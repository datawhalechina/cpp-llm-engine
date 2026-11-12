// DeepSeek2 CUDA Model：实现 Host 路由、设备执行的 MoE 正确性路径。
#include "cpp_llm_engine/model/cuda_deepseek2_moe_feed_forward.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "real_layer_weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>

namespace cpp_llm_engine::model {
namespace {

core::WeightFormat format_of(chapter01::DType dtype) {
    if (dtype == chapter01::DType::Q2_K) return core::WeightFormat::Q2K;
    if (dtype == chapter01::DType::Q3_K) return core::WeightFormat::Q3K;
    if (dtype == chapter01::DType::IQ4_NL) return core::WeightFormat::IQ4_NL;
    return core::WeightFormat::Unknown;
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t value = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            std::uint32_t normalized = mantissa;
            std::uint32_t adjusted = 113;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --adjusted;
            }
            value = sign | (adjusted << 23) |
                ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float converted = 0.0F;
    std::memcpy(&converted, &value, sizeof(converted));
    return converted;
}

core::Status read_router(const chapter01::Tensor& tensor,
                         std::vector<float>* output) {
    if (output == nullptr || tensor.shape().size() != 2 ||
        tensor.raw_data() == nullptr) {
        return core::Status::error("DeepSeek2 CUDA Router tensor is invalid");
    }
    std::vector<float> values(tensor.numel());
    if (tensor.dtype() == chapter01::DType::Float32) {
        std::memcpy(values.data(), tensor.raw_data(), tensor.bytes());
    } else if (tensor.dtype() == chapter01::DType::Float16) {
        const auto* source =
            reinterpret_cast<const std::uint16_t*>(tensor.raw_data());
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = half_to_float(source[index]);
        }
    } else {
        return core::Status::error(
            "DeepSeek2 CUDA Router requires Float32 or Float16");
    }
    *output = std::move(values);
    return core::Status::ok();
}

core::Status upload_matrix(cuda::CudaExecutionContext* context,
                           const void* data,
                           std::size_t bytes,
                           std::size_t rows,
                           std::size_t cols,
                           core::WeightFormat format,
                           cuda::CudaQuantizedMatrix* output) {
    core::Status status = cuda::CudaQuantizedMatrix::create(
        rows, cols, format, output);
    if (!status.is_ok()) return status;
    return context->upload(output, data, bytes);
}

core::Status upload_norm(cuda::CudaExecutionContext* context,
                         const chapter01::Tensor& tensor,
                         cuda::CudaTensor* output) {
    if (tensor.dtype() != chapter01::DType::Float32 ||
        tensor.shape().size() != 1 || tensor.shape()[0] <= 0 ||
        tensor.raw_data() == nullptr) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE norm must be a Float32 vector");
    }
    core::Status status = cuda::CudaTensor::create_contiguous(
        {static_cast<std::size_t>(tensor.shape()[0])},
        core::DType::Float32, output);
    if (!status.is_ok()) return status;
    return context->upload(output, tensor.raw_data(), tensor.bytes());
}

core::Status upload_linear(cuda::CudaExecutionContext* context,
                           const chapter01::Tensor& tensor,
                           cuda::CudaQuantizedMatrix* output) {
    if (tensor.shape().size() != 2 || tensor.shape()[0] <= 0 ||
        tensor.shape()[1] <= 0 || tensor.raw_data() == nullptr) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE Linear tensor is invalid");
    }
    const core::WeightFormat format = format_of(tensor.dtype());
    if (format == core::WeightFormat::Unknown) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE Linear format is unsupported");
    }
    return upload_matrix(context, tensor.raw_data(), tensor.bytes(),
        static_cast<std::size_t>(tensor.shape()[1]),
        static_cast<std::size_t>(tensor.shape()[0]), format, output);
}

core::Status upload_shared(cuda::CudaExecutionContext* context,
                           const chapter01::Tensor& tensor,
                           cuda::CudaQuantizedMatrix* output) {
    if (tensor.shape().size() != 2 || tensor.shape()[0] <= 0 ||
        tensor.shape()[1] <= 0 || tensor.raw_data() == nullptr) {
        return core::Status::error("DeepSeek2 CUDA shared expert is invalid");
    }
    const core::WeightFormat format = format_of(tensor.dtype());
    if (format == core::WeightFormat::Unknown) {
        return core::Status::error(
            "DeepSeek2 CUDA shared expert format is unsupported");
    }
    return upload_matrix(context, tensor.raw_data(), tensor.bytes(),
        static_cast<std::size_t>(tensor.shape()[1]),
        static_cast<std::size_t>(tensor.shape()[0]), format, output);
}

core::Status upload_experts(
    cuda::CudaExecutionContext* context,
    const chapter01::Tensor& tensor,
    std::vector<cuda::CudaQuantizedMatrix>* output) {
    if (output == nullptr || tensor.shape().size() != 3 ||
        tensor.shape()[0] <= 0 || tensor.shape()[1] <= 0 ||
        tensor.shape()[2] <= 0 || tensor.raw_data() == nullptr) {
        return core::Status::error("DeepSeek2 CUDA packed experts are invalid");
    }
    const core::WeightFormat format = format_of(tensor.dtype());
    if (format == core::WeightFormat::Unknown) {
        return core::Status::error(
            "DeepSeek2 CUDA packed expert format is unsupported");
    }
    const auto cols = static_cast<std::size_t>(tensor.shape()[0]);
    const auto rows = static_cast<std::size_t>(tensor.shape()[1]);
    const auto count = static_cast<std::size_t>(tensor.shape()[2]);
    if (tensor.bytes() % count != 0) {
        return core::Status::error(
            "DeepSeek2 CUDA packed expert payload is not divisible");
    }
    const std::size_t bytes = tensor.bytes() / count;
    std::vector<cuda::CudaQuantizedMatrix> matrices(count);
    const auto* source = reinterpret_cast<const std::uint8_t*>(
        tensor.raw_data());
    for (std::size_t expert = 0; expert < count; ++expert) {
        core::Status status = upload_matrix(
            context, source + expert * bytes, bytes, rows, cols, format,
            &matrices[expert]);
        if (!status.is_ok()) return status;
    }
    *output = std::move(matrices);
    return core::Status::ok();
}

core::Status run_expert(cuda::CudaExecutionContext* context,
                        const cuda::CudaTensor& input,
                        const cuda::CudaQuantizedMatrix& gate_weights,
                        const cuda::CudaQuantizedMatrix& up_weights,
                        const cuda::CudaQuantizedMatrix& down_weights,
                        cuda::CudaTensor* gate,
                        cuda::CudaTensor* up,
                        cuda::CudaTensor* activated,
                        cuda::CudaTensor* output) {
    core::Status status = context->quantized_linear(
        gate_weights, input, gate);
    if (status.is_ok()) status = context->quantized_linear(
        up_weights, input, up);
    if (status.is_ok()) status = context->swiglu(*gate, *up, activated);
    if (status.is_ok()) status = context->quantized_linear(
        down_weights, *activated, output);
    return status;
}

core::Status build_route(
    const CudaDeepSeek2MoeConfig& config,
    const std::vector<float>& router,
    const std::vector<float>& hidden,
    CudaDeepSeek2MoeRoute* route) {
    std::vector<float> logits(config.expert_count, 0.0F);
    for (std::size_t expert = 0; expert < config.expert_count; ++expert) {
        for (std::size_t column = 0; column < config.hidden_size; ++column) {
            logits[expert] += router[expert * config.hidden_size + column] *
                hidden[column];
        }
    }
    std::vector<std::size_t> order(config.expert_count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
        [&logits](std::size_t left, std::size_t right) {
            if (logits[left] == logits[right]) return left < right;
            return logits[left] > logits[right];
        });
    order.resize(config.top_k);
    const float maximum = *std::max_element(logits.begin(), logits.end());
    float denominator = 0.0F;
    for (float value : logits) denominator += std::exp(value - maximum);
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
        return core::Status::error("DeepSeek2 CUDA MoE route is not finite");
    }
    std::vector<float> route_weights(config.top_k);
    float selected_sum = 0.0F;
    for (std::size_t index = 0; index < config.top_k; ++index) {
        route_weights[index] =
            std::exp(logits[order[index]] - maximum) / denominator;
        selected_sum += route_weights[index];
    }
    if (config.normalize_weights) {
        for (float& value : route_weights) value /= selected_sum;
    }
    for (float& value : route_weights) value *= config.weight_scale;
    *route = {std::move(order), std::move(route_weights)};
    return core::Status::ok();
}

struct BatchLinearWorkspace final {
    cuda::CudaTensor input;
    cuda::CudaTensor gate;
    cuda::CudaTensor up;
    cuda::CudaTensor activated;
    cuda::CudaTensor output;
};

core::Status create_batch_linear_workspace(
    std::size_t rows,
    std::size_t hidden_size,
    std::size_t expert_hidden_size,
    BatchLinearWorkspace* output) {
    BatchLinearWorkspace replacement;
    core::Status status = cuda::CudaTensor::create_contiguous(
        {rows, hidden_size}, core::DType::Float32, &replacement.input);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {rows, expert_hidden_size}, core::DType::Float32, &replacement.gate);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {rows, expert_hidden_size}, core::DType::Float32, &replacement.up);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {rows, expert_hidden_size}, core::DType::Float32,
        &replacement.activated);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {rows, hidden_size}, core::DType::Float32, &replacement.output);
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status run_expert_batch(
    cuda::CudaExecutionContext* context,
    const std::vector<const cuda::CudaTensor*>& input_rows,
    const cuda::CudaQuantizedMatrix& gate_weights,
    const cuda::CudaQuantizedMatrix& up_weights,
    const cuda::CudaQuantizedMatrix& down_weights,
    BatchLinearWorkspace* workspace,
    std::vector<float>* output) {
    core::Status status = context->pack_rows(input_rows, &workspace->input);
    if (status.is_ok()) status = context->quantized_linear_batch(
        gate_weights, workspace->input, &workspace->gate);
    if (status.is_ok()) status = context->quantized_linear_batch(
        up_weights, workspace->input, &workspace->up);
    if (status.is_ok()) status = context->swiglu(
        workspace->gate, workspace->up, &workspace->activated);
    if (status.is_ok()) status = context->quantized_linear_batch(
        down_weights, workspace->activated, &workspace->output);
    output->resize(workspace->output.size_bytes() / sizeof(float));
    if (status.is_ok()) status = context->download(
        workspace->output, output->data(), workspace->output.size_bytes());
    if (status.is_ok()) status = context->synchronize();
    return status;
}

}  // namespace

core::Status CudaDeepSeek2MoeConfig::validate() const {
    if (hidden_size == 0 || expert_hidden_size == 0 ||
        shared_expert_hidden_size == 0 || expert_count == 0 ||
        top_k == 0 || top_k > expert_count || !std::isfinite(weight_scale) ||
        weight_scale <= 0.0F) {
        return core::Status::error("DeepSeek2 CUDA MoE config is invalid");
    }
    return core::Status::ok();
}

core::Status CudaDeepSeek2MoeWeights::upload(
    cuda::CudaExecutionContext* context,
    const chapter01::RealDeepSeek2MoeLayerWeights& host,
    CudaDeepSeek2MoeWeights* output) {
    if (context == nullptr || output == nullptr || !host.valid()) {
        return core::Status::error("DeepSeek2 CUDA MoE upload input is invalid");
    }
    CudaDeepSeek2MoeWeights replacement;
    core::Status status = read_router(host.router, &replacement.router_);
    if (status.is_ok()) status = upload_norm(
        context, host.attn_norm, &replacement.attention_norm_);
    if (status.is_ok()) status = upload_linear(
        context, host.attn_q, &replacement.query_);
    if (status.is_ok()) status = upload_linear(
        context, host.attn_kv_a_mqa, &replacement.kv_a_mqa_);
    if (status.is_ok()) status = upload_norm(
        context, host.attn_kv_a_norm, &replacement.kv_a_norm_);
    if (status.is_ok()) status = upload_linear(
        context, host.attn_kv_b, &replacement.kv_b_);
    if (status.is_ok()) {
        status = cuda::CudaTensor::create_contiguous(
            {replacement.kv_b_.rows(), replacement.kv_b_.cols()},
            core::DType::Float32, &replacement.kv_b_dequantized_);
    }
    if (status.is_ok()) status = context->dequantize(
        replacement.kv_b_, &replacement.kv_b_dequantized_);
    if (status.is_ok()) status = upload_linear(
        context, host.attn_output, &replacement.attention_output_);
    if (status.is_ok()) status = upload_norm(
        context, host.ffn_norm, &replacement.ffn_norm_);
    if (status.is_ok()) status = upload_experts(
        context, host.expert_gate, &replacement.expert_gate_);
    if (status.is_ok()) status = upload_experts(
        context, host.expert_up, &replacement.expert_up_);
    if (status.is_ok()) status = upload_experts(
        context, host.expert_down, &replacement.expert_down_);
    if (status.is_ok()) status = upload_shared(
        context, host.shared_expert_gate, &replacement.shared_gate_);
    if (status.is_ok()) status = upload_shared(
        context, host.shared_expert_up, &replacement.shared_up_);
    if (status.is_ok()) status = upload_shared(
        context, host.shared_expert_down, &replacement.shared_down_);
    if (!status.is_ok()) return status;
    status = context->synchronize();
    if (!status.is_ok()) return status;
    replacement.hidden_size_ = replacement.expert_gate_.front().cols();
    replacement.expert_hidden_size_ = replacement.expert_gate_.front().rows();
    replacement.shared_expert_hidden_size_ = replacement.shared_gate_.rows();
    if (!replacement.valid()) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE uploaded weights are inconsistent");
    }
    *output = std::move(replacement);
    return core::Status::ok();
}

bool CudaDeepSeek2MoeWeights::valid() const {
    if (hidden_size_ == 0 || expert_hidden_size_ == 0 ||
        shared_expert_hidden_size_ == 0 ||
        expert_gate_.empty() || expert_gate_.size() != expert_up_.size() ||
        expert_gate_.size() != expert_down_.size() ||
        router_.size() != hidden_size_ * expert_gate_.size() ||
        shared_gate_.cols() != hidden_size_ ||
        shared_gate_.rows() != shared_expert_hidden_size_ ||
        shared_up_.cols() != hidden_size_ ||
        shared_up_.rows() != shared_expert_hidden_size_ ||
        shared_down_.cols() != shared_expert_hidden_size_ ||
        shared_down_.rows() != hidden_size_ || attention_norm_.empty() ||
        query_.empty() || kv_a_mqa_.empty() || kv_a_norm_.empty() ||
        kv_b_dequantized_.empty() || attention_output_.empty() ||
        ffn_norm_.empty()) return false;
    for (std::size_t index = 0; index < expert_gate_.size(); ++index) {
        if (expert_gate_[index].cols() != hidden_size_ ||
            expert_gate_[index].rows() != expert_hidden_size_ ||
            expert_up_[index].cols() != hidden_size_ ||
            expert_up_[index].rows() != expert_hidden_size_ ||
            expert_down_[index].cols() != expert_hidden_size_ ||
            expert_down_[index].rows() != hidden_size_) return false;
    }
    return true;
}

std::size_t CudaDeepSeek2MoeWeights::hidden_size() const { return hidden_size_; }
std::size_t CudaDeepSeek2MoeWeights::expert_hidden_size() const {
    return expert_hidden_size_;
}
std::size_t CudaDeepSeek2MoeWeights::shared_expert_hidden_size() const {
    return shared_expert_hidden_size_;
}
std::size_t CudaDeepSeek2MoeWeights::expert_count() const {
    return expert_gate_.size();
}
const std::vector<float>& CudaDeepSeek2MoeWeights::router() const {
    return router_;
}
const cuda::CudaQuantizedMatrix& CudaDeepSeek2MoeWeights::expert_gate(
    std::size_t index) const { return expert_gate_.at(index); }
const cuda::CudaQuantizedMatrix& CudaDeepSeek2MoeWeights::expert_up(
    std::size_t index) const { return expert_up_.at(index); }
const cuda::CudaQuantizedMatrix& CudaDeepSeek2MoeWeights::expert_down(
    std::size_t index) const { return expert_down_.at(index); }
const cuda::CudaQuantizedMatrix& CudaDeepSeek2MoeWeights::shared_gate() const {
    return shared_gate_;
}
const cuda::CudaQuantizedMatrix& CudaDeepSeek2MoeWeights::shared_up() const {
    return shared_up_;
}
const cuda::CudaQuantizedMatrix& CudaDeepSeek2MoeWeights::shared_down() const {
    return shared_down_;
}
const cuda::CudaTensor& CudaDeepSeek2MoeWeights::attention_norm() const {
    return attention_norm_;
}
cuda::CudaLinearWeight CudaDeepSeek2MoeWeights::query() const {
    return cuda::CudaLinearWeight::quantized(&query_);
}
cuda::CudaLinearWeight CudaDeepSeek2MoeWeights::kv_a_mqa() const {
    return cuda::CudaLinearWeight::quantized(&kv_a_mqa_);
}
const cuda::CudaTensor& CudaDeepSeek2MoeWeights::kv_a_norm() const {
    return kv_a_norm_;
}
const cuda::CudaTensor& CudaDeepSeek2MoeWeights::kv_b_dequantized() const {
    return kv_b_dequantized_;
}
cuda::CudaLinearWeight CudaDeepSeek2MoeWeights::attention_output() const {
    return cuda::CudaLinearWeight::quantized(&attention_output_);
}
const cuda::CudaTensor& CudaDeepSeek2MoeWeights::ffn_norm() const {
    return ffn_norm_;
}

core::Status CudaDeepSeek2MoeFeedForward::create(
    const CudaDeepSeek2MoeConfig& config,
    CudaDeepSeek2MoeFeedForward* output) {
    if (output == nullptr || !config.validate().is_ok()) {
        return core::Status::error("DeepSeek2 CUDA MoE output/config is invalid");
    }
    CudaDeepSeek2MoeFeedForward replacement;
    replacement.config_ = config;
    core::Status status = cuda::CudaTensor::create_contiguous(
        {config.expert_hidden_size}, core::DType::Float32,
        &replacement.gate_);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {config.expert_hidden_size}, core::DType::Float32,
        &replacement.up_);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {config.expert_hidden_size}, core::DType::Float32,
        &replacement.activated_);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {config.hidden_size}, core::DType::Float32,
        &replacement.expert_output_);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {config.shared_expert_hidden_size}, core::DType::Float32,
        &replacement.shared_gate_);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {config.shared_expert_hidden_size}, core::DType::Float32,
        &replacement.shared_up_);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {config.shared_expert_hidden_size}, core::DType::Float32,
        &replacement.shared_activated_);
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaDeepSeek2MoeFeedForward::forward(
    cuda::CudaExecutionContext* context,
    const CudaDeepSeek2MoeWeights& weights,
    const cuda::CudaTensor& normalized_hidden,
    cuda::CudaTensor* output,
    CudaDeepSeek2MoeRoute* route) {
    if (context == nullptr || output == nullptr || route == nullptr ||
        !weights.valid() || weights.hidden_size() != config_.hidden_size ||
        weights.expert_hidden_size() != config_.expert_hidden_size ||
        weights.shared_expert_hidden_size() !=
            config_.shared_expert_hidden_size ||
        weights.expert_count() != config_.expert_count ||
        normalized_hidden.contract().shape.dimensions !=
            std::vector<std::size_t>{config_.hidden_size} ||
        output->contract().shape.dimensions !=
            std::vector<std::size_t>{config_.hidden_size}) {
        return core::Status::error("DeepSeek2 CUDA MoE forward input is invalid");
    }
    std::vector<float> hidden(config_.hidden_size);
    core::Status status = context->download(
        normalized_hidden, hidden.data(), hidden.size() * sizeof(float));
    if (status.is_ok()) status = context->synchronize();
    if (!status.is_ok()) return status;

    CudaDeepSeek2MoeRoute built_route;
    status = build_route(config_, weights.router(), hidden, &built_route);
    if (!status.is_ok()) return status;

    std::vector<float> built(config_.hidden_size, 0.0F);
    std::vector<float> expert_host(config_.hidden_size);
    for (std::size_t selected = 0;
         selected < built_route.expert_indices.size(); ++selected) {
        const std::size_t expert = built_route.expert_indices[selected];
        status = run_expert(context, normalized_hidden,
            weights.expert_gate(expert), weights.expert_up(expert),
            weights.expert_down(expert), &gate_, &up_, &activated_,
            &expert_output_);
        if (status.is_ok()) status = context->download(
            expert_output_, expert_host.data(), expert_host.size() * sizeof(float));
        if (status.is_ok()) status = context->synchronize();
        if (!status.is_ok()) return status;
        for (std::size_t column = 0; column < built.size(); ++column) {
            built[column] += built_route.weights[selected] * expert_host[column];
        }
    }
    status = run_expert(context, normalized_hidden, weights.shared_gate(),
        weights.shared_up(), weights.shared_down(), &shared_gate_, &shared_up_,
        &shared_activated_, &expert_output_);
    if (status.is_ok()) status = context->download(
        expert_output_, expert_host.data(), expert_host.size() * sizeof(float));
    if (status.is_ok()) status = context->synchronize();
    if (!status.is_ok()) return status;
    for (std::size_t column = 0; column < built.size(); ++column) {
        built[column] += expert_host[column];
        if (!std::isfinite(built[column])) {
            return core::Status::error(
                "DeepSeek2 CUDA MoE output is not finite");
        }
    }
    status = context->upload(output, built.data(), built.size() * sizeof(float));
    if (!status.is_ok()) return status;
    *route = std::move(built_route);
    return core::Status::ok();
}

struct CudaDeepSeek2MoeBatchWorkspace::Impl final {
    CudaDeepSeek2MoeConfig config;
    std::size_t batch_size = 0;
    std::size_t use_count = 0;
    std::size_t active_expert_groups = 0;
    std::size_t multi_row_expert_groups = 0;
    std::size_t routed_rows = 0;
    std::vector<BatchLinearWorkspace> expert_groups;
    BatchLinearWorkspace shared;
};

CudaDeepSeek2MoeBatchWorkspace::CudaDeepSeek2MoeBatchWorkspace() = default;
CudaDeepSeek2MoeBatchWorkspace::CudaDeepSeek2MoeBatchWorkspace(
    std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaDeepSeek2MoeBatchWorkspace::~CudaDeepSeek2MoeBatchWorkspace() = default;
CudaDeepSeek2MoeBatchWorkspace::CudaDeepSeek2MoeBatchWorkspace(
    CudaDeepSeek2MoeBatchWorkspace&&) noexcept = default;
CudaDeepSeek2MoeBatchWorkspace& CudaDeepSeek2MoeBatchWorkspace::operator=(
    CudaDeepSeek2MoeBatchWorkspace&&) noexcept = default;

core::Status CudaDeepSeek2MoeBatchWorkspace::create(
    const CudaDeepSeek2MoeConfig& config,
    std::size_t batch_size,
    CudaDeepSeek2MoeBatchWorkspace* output) {
    if (output == nullptr || batch_size == 0 || !config.validate().is_ok()) {
        return core::Status::error(
            "DeepSeek2 CUDA MoE batch config is invalid");
    }
    auto impl = std::make_unique<Impl>();
    impl->config = config;
    impl->batch_size = batch_size;
    impl->expert_groups.resize(batch_size);
    core::Status status = core::Status::ok();
    for (std::size_t rows = 1; status.is_ok() && rows <= batch_size; ++rows) {
        status = create_batch_linear_workspace(
            rows, config.hidden_size, config.expert_hidden_size,
            &impl->expert_groups[rows - 1]);
    }
    if (status.is_ok()) status = create_batch_linear_workspace(
        batch_size, config.hidden_size, config.shared_expert_hidden_size,
        &impl->shared);
    if (!status.is_ok()) return status;
    *output = CudaDeepSeek2MoeBatchWorkspace(std::move(impl));
    return core::Status::ok();
}

core::Status CudaDeepSeek2MoeBatchWorkspace::forward(
    cuda::CudaExecutionContext* context,
    const CudaDeepSeek2MoeWeights& weights,
    const std::vector<const cuda::CudaTensor*>& normalized_hidden,
    const std::vector<cuda::CudaTensor*>& output,
    std::vector<CudaDeepSeek2MoeRoute>* routes) {
    if (impl_ == nullptr || context == nullptr || routes == nullptr ||
        normalized_hidden.size() != impl_->batch_size ||
        output.size() != impl_->batch_size || !weights.valid()) {
        return core::Status::error("DeepSeek2 CUDA MoE batch input is invalid");
    }
    for (std::size_t row = 0; row < impl_->batch_size; ++row) {
        if (normalized_hidden[row] == nullptr || output[row] == nullptr ||
            normalized_hidden[row]->contract().shape.dimensions !=
                std::vector<std::size_t>{impl_->config.hidden_size} ||
            output[row]->contract().shape.dimensions !=
                std::vector<std::size_t>{impl_->config.hidden_size}) {
            return core::Status::error(
                "DeepSeek2 CUDA MoE batch row is invalid");
        }
    }

    std::vector<std::vector<float>> hidden(
        impl_->batch_size, std::vector<float>(impl_->config.hidden_size));
    core::Status status = core::Status::ok();
    for (std::size_t row = 0; status.is_ok() && row < impl_->batch_size; ++row) {
        status = context->download(
            *normalized_hidden[row], hidden[row].data(),
            hidden[row].size() * sizeof(float));
    }
    if (status.is_ok()) status = context->synchronize();
    if (!status.is_ok()) return status;

    std::vector<CudaDeepSeek2MoeRoute> built_routes(impl_->batch_size);
    for (std::size_t row = 0; row < impl_->batch_size; ++row) {
        status = build_route(
            impl_->config, weights.router(), hidden[row], &built_routes[row]);
        if (!status.is_ok()) return status;
    }
    struct Assignment { std::size_t row; std::size_t slot; };
    std::vector<std::vector<Assignment>> assignments(impl_->config.expert_count);
    for (std::size_t row = 0; row < built_routes.size(); ++row) {
        for (std::size_t slot = 0;
             slot < built_routes[row].expert_indices.size(); ++slot) {
            assignments[built_routes[row].expert_indices[slot]].push_back(
                {row, slot});
        }
    }

    std::vector<std::vector<float>> built(
        impl_->batch_size, std::vector<float>(impl_->config.hidden_size, 0.0F));
    std::size_t active_expert_groups = 0;
    std::size_t multi_row_expert_groups = 0;
    for (std::size_t expert = 0; expert < assignments.size(); ++expert) {
        const auto& group = assignments[expert];
        if (group.empty()) continue;
        ++active_expert_groups;
        if (group.size() > 1) ++multi_row_expert_groups;
        std::vector<const cuda::CudaTensor*> rows;
        rows.reserve(group.size());
        for (const auto assignment : group) {
            rows.push_back(normalized_hidden[assignment.row]);
        }
        std::vector<float> flat;
        status = run_expert_batch(
            context, rows, weights.expert_gate(expert),
            weights.expert_up(expert), weights.expert_down(expert),
            &impl_->expert_groups[group.size() - 1], &flat);
        if (!status.is_ok()) return status;
        for (std::size_t item = 0; item < group.size(); ++item) {
            const auto assignment = group[item];
            const float weight = built_routes[assignment.row].weights[
                assignment.slot];
            for (std::size_t column = 0;
                 column < impl_->config.hidden_size; ++column) {
                built[assignment.row][column] +=
                    weight * flat[item * impl_->config.hidden_size + column];
            }
        }
    }

    std::vector<float> shared;
    status = run_expert_batch(
        context, normalized_hidden, weights.shared_gate(), weights.shared_up(),
        weights.shared_down(), &impl_->shared, &shared);
    if (!status.is_ok()) return status;
    for (std::size_t row = 0; row < impl_->batch_size; ++row) {
        for (std::size_t column = 0;
             column < impl_->config.hidden_size; ++column) {
            built[row][column] +=
                shared[row * impl_->config.hidden_size + column];
            if (!std::isfinite(built[row][column])) {
                return core::Status::error(
                    "DeepSeek2 CUDA MoE batch output is not finite");
            }
        }
        status = context->upload(
            output[row], built[row].data(), built[row].size() * sizeof(float));
        if (!status.is_ok()) return status;
    }
    status = context->synchronize();
    if (!status.is_ok()) return status;
    *routes = std::move(built_routes);
    ++impl_->use_count;
    impl_->active_expert_groups += active_expert_groups;
    impl_->multi_row_expert_groups += multi_row_expert_groups;
    impl_->routed_rows += impl_->batch_size * impl_->config.top_k;
    return core::Status::ok();
}

std::size_t CudaDeepSeek2MoeBatchWorkspace::batch_size() const {
    return impl_ == nullptr ? 0 : impl_->batch_size;
}
std::size_t CudaDeepSeek2MoeBatchWorkspace::use_count() const {
    return impl_ == nullptr ? 0 : impl_->use_count;
}
std::size_t CudaDeepSeek2MoeBatchWorkspace::active_expert_groups() const {
    return impl_ == nullptr ? 0 : impl_->active_expert_groups;
}
std::size_t CudaDeepSeek2MoeBatchWorkspace::multi_row_expert_groups() const {
    return impl_ == nullptr ? 0 : impl_->multi_row_expert_groups;
}
std::size_t CudaDeepSeek2MoeBatchWorkspace::routed_rows() const {
    return impl_ == nullptr ? 0 : impl_->routed_rows;
}

}  // namespace cpp_llm_engine::model
