// Core：公共接口的独立回归测试
// 覆盖 Tensor 布局、量化字节数、Decoder 结构和默认批处理失败路径。
// 失败时输出用例、表达式和源码位置。
// 验证：Core CTest、默认回归和 Release 契约回归。
#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/core/decoder_contract.h"
#include "cpp_llm_engine/core/decoder_elementwise_contract.h"
#include "cpp_llm_engine/core/gqa_attention_contract.h"
#include "cpp_llm_engine/core/linear_contract.h"
#include "cpp_llm_engine/core/operator.h"
#include "cpp_llm_engine/core/quantization_contract.h"
#include "cpp_llm_engine/core/rope_contract.h"
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/core/tensor_view_ref.h"
#include "cpp_llm_engine/testing/test.h"

#include <limits>
#include <string_view>

namespace {

class CpuBackend final : public cpp_llm_engine::core::Backend {
public:
    cpp_llm_engine::core::BackendKind kind() const override {
        return cpp_llm_engine::core::BackendKind::CPU;
    }
    std::string_view name() const override { return "cpu-reference"; }
};

class MatmulOperator final : public cpp_llm_engine::core::Operator {
public:
    std::string_view name() const override { return "matmul"; }
};

class CopyLinearBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override {
        return cpp_llm_engine::core::BackendKind::CPU;
    }
    std::string_view name() const override { return "copy-linear"; }

    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override {
        if (request.input == nullptr || request.output == nullptr ||
            request.input_size != request.cols || request.output_size < request.rows ||
            request.rows != request.cols) {
            return cpp_llm_engine::core::Status::error("invalid copy Linear request");
        }
        for (std::size_t index = 0; index < request.rows; ++index) {
            request.output[index] = request.input[index];
        }
        return cpp_llm_engine::core::Status::ok();
    }
};

}  // namespace

int main() {
    using namespace cpp_llm_engine::core;
    cpp_llm_engine::testing::TestSuite suite("core_contracts");

    suite.run("shape and contiguous tensor contract", [&] {
        Shape shape{{2, 3, 4}};
        std::size_t elements = 0;
        CPP_LLM_CHECK(suite, shape.checked_numel(&elements).is_ok());
        CPP_LLM_CHECK(suite, elements == 24);

        Shape overflow{{std::numeric_limits<std::size_t>::max(), 2}};
        CPP_LLM_CHECK(suite, !overflow.checked_numel(&elements).is_ok());

        const TensorContract tensor{
            shape, {12, 4, 1}, DType::Float32, DeviceKind::CPU, 0,
            24 * sizeof(float)};
        CPP_LLM_CHECK(suite, tensor.device == DeviceKind::CPU);
        CPP_LLM_CHECK(suite, tensor.is_contiguous_row_major());
        CPP_LLM_CHECK(suite, tensor.validate_element_view(sizeof(float)).is_ok());

        const TensorContract singleton_stride{
            Shape{{2, 1, 4}}, {4, 99, 1}, DType::Float32,
            DeviceKind::CPU, 0, 8 * sizeof(float)};
        CPP_LLM_CHECK(suite, singleton_stride.is_contiguous_row_major());
        auto transposed_tensor = tensor;
        transposed_tensor.shape.dimensions = {3, 2, 4};
        transposed_tensor.strides = {4, 12, 1};
        CPP_LLM_CHECK(suite, !transposed_tensor.is_contiguous_row_major());
    });

    suite.run("tensor view validates storage and offset", [&] {
        const TensorContract offset_view{
            Shape{{2, 2}}, {3, 1}, DType::Float32, DeviceKind::CPU, 1,
            6 * sizeof(float)};
        std::size_t required_elements = 0;
        CPP_LLM_CHECK(
            suite, offset_view.checked_required_elements(&required_elements).is_ok());
        CPP_LLM_CHECK(suite, required_elements == 6);
        CPP_LLM_CHECK(suite, offset_view.validate_element_view(sizeof(float)).is_ok());
        float view_storage[6]{};
        const TensorViewRef tensor_view{view_storage, offset_view};
        CPP_LLM_CHECK(suite, tensor_view.validate(sizeof(float)).is_ok());
        const TensorViewRef null_view{nullptr, offset_view};
        CPP_LLM_CHECK(suite, !null_view.validate(sizeof(float)).is_ok());
        MutableTensorViewRef mutable_view{view_storage, offset_view};
        CPP_LLM_CHECK(suite, mutable_view.validate(sizeof(float)).is_ok());
        static_cast<float*>(mutable_view.storage_data)[
            mutable_view.contract.storage_offset_elements] = 3.5F;
        CPP_LLM_CHECK(suite, view_storage[1] == 3.5F);
        CPP_LLM_CHECK(suite, mutable_view.as_const().validate(sizeof(float)).is_ok());
        const MutableTensorViewRef null_mutable_view{nullptr, offset_view};
        CPP_LLM_CHECK(suite, !null_mutable_view.validate(sizeof(float)).is_ok());

        const TensorContract short_storage{
            Shape{{2, 2}}, {3, 1}, DType::Float32, DeviceKind::CPU, 1,
            5 * sizeof(float)};
        CPP_LLM_CHECK(
            suite, !short_storage.validate_element_view(sizeof(float)).is_ok());
        const MutableTensorViewRef short_mutable_view{view_storage, short_storage};
        CPP_LLM_CHECK(suite, !short_mutable_view.validate(sizeof(float)).is_ok());
        const TensorContract bad_rank{
            Shape{{2, 2}}, {2}, DType::Float32, DeviceKind::CPU, 0,
            4 * sizeof(float)};
        CPP_LLM_CHECK(suite, !bad_rank.validate_element_view(sizeof(float)).is_ok());
    });

    suite.run("quantized matrix byte contract", [&] {
        CPP_LLM_CHECK(suite, static_cast<int>(DType::Q6K) > static_cast<int>(DType::Q4K));
        std::size_t quantized_bytes = 0;
        CPP_LLM_CHECK(suite, checked_quantized_matrix_bytes(
            WeightFormat::Q4K, 2, 512, &quantized_bytes).is_ok());
        CPP_LLM_CHECK(suite, quantized_bytes == 4 * kQ4KBlockBytes);
        CPP_LLM_CHECK(suite, checked_quantized_matrix_bytes(
            WeightFormat::Q6K, 3, 256, &quantized_bytes).is_ok());
        CPP_LLM_CHECK(suite, quantized_bytes == 3 * kQ6KBlockBytes);
        CPP_LLM_CHECK(suite, checked_quantized_matrix_bytes(
            WeightFormat::Q8_0, 2, 96, &quantized_bytes).is_ok());
        CPP_LLM_CHECK(suite, quantized_bytes == 6 * kQ8_0BlockBytes);
        CPP_LLM_CHECK(suite, !checked_quantized_matrix_bytes(
            WeightFormat::Float32, 1, 256, &quantized_bytes).is_ok());
        CPP_LLM_CHECK(suite, !checked_quantized_matrix_bytes(
            WeightFormat::Q4K, 1, 255, &quantized_bytes).is_ok());
        CPP_LLM_CHECK(suite, !checked_quantized_matrix_bytes(
            WeightFormat::Q4K,
            std::numeric_limits<std::size_t>::max(),
            256,
            &quantized_bytes).is_ok());
        CPP_LLM_CHECK(suite, Status::ok().is_ok());
        CPP_LLM_CHECK(suite, !Status::error("invalid shape").is_ok());
    });

    suite.run("decoder rope and gqa contracts", [&] {
        const DecoderLayerContract decoder_layer{
            2048, 5632, 32, 4, 64, 64,
            DecoderNormalizationKind::RmsNorm,
            DecoderActivationKind::SwiGlu,
            DecoderPositionEncodingKind::Rotary,
            true, true};
        CPP_LLM_CHECK(suite, decoder_layer.validate().is_ok());
        const DecoderModelContract decoder_model{32000, 22, 2048, decoder_layer};
        CPP_LLM_CHECK(suite, decoder_model.validate().is_ok());
        const DecoderEmbeddingSequence embeddings{
            2, 2048, std::vector<float>(4096, 0.0f)};
        CPP_LLM_CHECK(suite, embeddings.validate(2048).is_ok());
        CPP_LLM_CHECK(suite, !embeddings.validate(960).is_ok());
        auto invalid_embeddings = embeddings;
        invalid_embeddings.values.pop_back();
        CPP_LLM_CHECK(suite, !invalid_embeddings.validate(2048).is_ok());
        auto invalid_heads = decoder_layer;
        invalid_heads.kv_head_count = 3;
        CPP_LLM_CHECK(suite, !invalid_heads.validate().is_ok());
        auto invalid_rope = decoder_layer;
        invalid_rope.position_dimension = 63;
        CPP_LLM_CHECK(suite, !invalid_rope.validate().is_ok());

        float rope_values[16]{};
        const RotaryEmbeddingRequest rope_request{
            rope_values, 1, 2, 8, 8, 0, 10000.0F};
        CPP_LLM_CHECK(suite, rope_request.validate().is_ok());
        auto odd_rope = rope_request;
        odd_rope.rotary_dimension = 7;
        CPP_LLM_CHECK(suite, !odd_rope.validate().is_ok());
        auto overflow_rope = rope_request;
        overflow_rope.token_count = 2;
        overflow_rope.position_offset = std::numeric_limits<std::size_t>::max();
        CPP_LLM_CHECK(suite, !overflow_rope.validate().is_ok());

        float query[16]{};
        float key[8]{};
        float value[8]{};
        float attention_output[16]{};
        const GqaAttentionRequest gqa_request{
            query, key, value, attention_output, 1, 1, 4, 2, 4, 0, true, 0.0F};
        CPP_LLM_CHECK(suite, gqa_request.validate().is_ok());
        CPP_LLM_CHECK(suite, gqa_request.score_count() == 4);
        auto invalid_gqa_heads = gqa_request;
        invalid_gqa_heads.query_head_count = 3;
        CPP_LLM_CHECK(suite, !invalid_gqa_heads.validate().is_ok());
        auto invalid_gqa_scale = gqa_request;
        invalid_gqa_scale.scale = -1.0F;
        CPP_LLM_CHECK(suite, !invalid_gqa_scale.validate().is_ok());

        float norm_input[4]{1.0F, 2.0F, 3.0F, 4.0F};
        float norm_weight[4]{1.0F, 1.0F, 1.0F, 1.0F};
        float elementwise_output[4]{};
        const RmsNormRequest rms_norm{
            norm_input, norm_weight, elementwise_output, 4, 1.0e-5F};
        CPP_LLM_CHECK(suite, rms_norm.validate().is_ok());
        auto invalid_epsilon = rms_norm;
        invalid_epsilon.epsilon = 0.0F;
        CPP_LLM_CHECK(suite, !invalid_epsilon.validate().is_ok());

        const SwiGluRequest swiglu{
            norm_input, norm_weight, elementwise_output, 4};
        CPP_LLM_CHECK(suite, swiglu.validate().is_ok());
        auto missing_gate = swiglu;
        missing_gate.gate = nullptr;
        CPP_LLM_CHECK(suite, !missing_gate.validate().is_ok());
    });

    suite.run("linear batch fallback validates before pointer arithmetic", [&] {
        CopyLinearBackend backend;
        float weights[4]{};
        float input[4]{1.0F, 2.0F, 3.0F, 4.0F};
        float output[4]{};
        const LinearBatchRequest request{
            weights, sizeof(weights), input, 4, output, 4, 2, 2, 2,
            WeightFormat::Float32};
        CPP_LLM_CHECK(suite, backend.compute_batch(request).is_ok());
        CPP_LLM_CHECK(suite, output[0] == 1.0F && output[3] == 4.0F);

        auto null_input = request;
        null_input.input = nullptr;
        CPP_LLM_CHECK(suite, !backend.compute_batch(null_input).is_ok());
        auto null_output = request;
        null_output.output = nullptr;
        CPP_LLM_CHECK(suite, !backend.compute_batch(null_output).is_ok());
        auto overflow = request;
        overflow.batch_size = std::numeric_limits<std::size_t>::max();
        CPP_LLM_CHECK(suite, !backend.compute_batch(overflow).is_ok());
    });

    suite.run("backend and operator identity", [&] {
        CpuBackend backend;
        MatmulOperator matmul;
        CPP_LLM_CHECK(suite, backend.kind() == BackendKind::CPU);
        CPP_LLM_CHECK(suite, backend.name() == "cpu-reference");
        CPP_LLM_CHECK(suite, matmul.name() == "matmul");
    });

    return suite.finish();
}
