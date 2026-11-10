// Model：实现 Qwen2MoE 最终 RMSNorm 与量化 LM Head。
#include "cpp_llm_engine/model/real_qwen2_moe_output.h"

#include "cpp_llm_engine/core/linear_contract.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cpp_llm_engine::model {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) error->assign(message);
    return false;
}

cpp_llm_engine::core::WeightFormat weight_format(chapter01::DType dtype) {
    using chapter01::DType;
    using cpp_llm_engine::core::WeightFormat;
    if (dtype == DType::Q3_K) return WeightFormat::Q3K;
    if (dtype == DType::Q4_K) return WeightFormat::Q4K;
    if (dtype == DType::Q5_0) return WeightFormat::Q5_0;
    if (dtype == DType::Q5_1) return WeightFormat::Q5_1;
    if (dtype == DType::Q5_K) return WeightFormat::Q5K;
    if (dtype == DType::Q6_K) return WeightFormat::Q6K;
    if (dtype == DType::Q8_0) return WeightFormat::Q8_0;
    return WeightFormat::Unknown;
}

}  // namespace

bool finish_qwen2_moe_output(
    const chapter01::ModelMetadata& metadata,
    const chapter01::Tensor& output_norm,
    const chapter01::Tensor& output_weight,
    const std::vector<float>& hidden,
    std::vector<float>* normalized_hidden,
    std::vector<float>* logits,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy) {
    if (normalized_hidden == nullptr || logits == nullptr ||
        hidden.size() != metadata.hidden_size ||
        output_norm.dtype() != chapter01::DType::Float32 ||
        output_norm.numel() != metadata.hidden_size ||
        output_norm.raw_data() == nullptr) {
        return fail(error, "Qwen2MoE output norm does not match hidden state");
    }
    const auto& shape = output_weight.shape();
    if (shape.size() != 2 || shape[0] !=
            static_cast<std::int64_t>(metadata.hidden_size) ||
        shape[1] != static_cast<std::int64_t>(metadata.vocab_size)) {
        return fail(error, "Qwen2MoE output weight does not match vocabulary");
    }
    const auto format = weight_format(output_weight.dtype());
    if (format == cpp_llm_engine::core::WeightFormat::Unknown) {
        return fail(error, "Qwen2MoE output weight format is not supported");
    }

    std::vector<float> scale(metadata.hidden_size, 0.0F);
    std::memcpy(scale.data(), output_norm.raw_data(),
                scale.size() * sizeof(float));
    double square_sum = 0.0;
    for (const float value : hidden) square_sum += value * value;
    const float inverse = 1.0F / std::sqrt(
        static_cast<float>(square_sum / hidden.size()) +
        metadata.rms_norm_epsilon);
    std::vector<float> normalized(hidden.size(), 0.0F);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        normalized[i] = hidden[i] * inverse * scale[i];
    }

    std::vector<float> output(metadata.vocab_size, 0.0F);
    const cpp_llm_engine::core::LinearRequest request{
        output_weight.raw_data(), output_weight.bytes(), normalized.data(),
        normalized.size(), output.data(), output.size(), metadata.vocab_size,
        metadata.hidden_size, format};
    const auto status =
        chapter05_quantization::compute_quantized_linear(request, policy);
    if (!status.is_ok()) return fail(error, status.message());
    if (!std::all_of(output.begin(), output.end(),
                     [](float value) { return std::isfinite(value); })) {
        return fail(error, "Qwen2MoE LM Head produced non-finite logits");
    }
    *normalized_hidden = std::move(normalized);
    *logits = std::move(output);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace cpp_llm_engine::model
