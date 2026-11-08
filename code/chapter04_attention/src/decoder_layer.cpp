// Chapter 04：DecoderLayer 组合实现
// 职责：按固定顺序连接 LayerNorm、Attention、Residual 和 FeedForward。
// 契约：任一组件未 ready 或输入 shape 不匹配时返回空 Tensor。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/decoder_layer.h"

#include <algorithm>
#include <utility>

namespace chapter04_attention {

DecoderLayer::DecoderLayer(std::size_t model_dim, std::size_t num_heads, std::size_t ffn_dim, std::size_t layer_index)
    : layer_index_(layer_index),
      pre_attn_norm_(model_dim),
      attention_(model_dim, num_heads),
      pre_ffn_norm_(model_dim),
      feed_forward_(model_dim, ffn_dim) {}

Tensor DecoderLayer::forward(const Tensor& input) const {
    if (!ready() || !input.ready()) {
        return {};
    }

    Tensor output(input.rows, input.cols);
    const auto status = try_forward(input.view_ref(), output.mutable_view_ref());
    if (!status.is_ok()) return {};
    return output;
}

cpp_llm_engine::core::Status DecoderLayer::try_forward(
    const cpp_llm_engine::core::TensorViewRef& input,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    using cpp_llm_engine::core::Status;

    if (!ready()) return Status::error("decoder layer is not ready");
    const auto input_status = validate_cpu_float32_matrix_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;
    const auto tokens = input.contract.shape.dimensions[0];
    const auto model_dim = attention_.model_dim();
    if (input.contract.shape.dimensions[1] != model_dim) {
        return Status::error("decoder layer input shape mismatch");
    }
    if (output.contract.shape.dimensions !=
        std::vector<std::size_t>{tokens, model_dim}) {
        return Status::error("decoder layer output shape mismatch");
    }

    Tensor attn_input(tokens, model_dim);
    auto status = pre_attn_norm_.try_forward(
        input, attn_input.mutable_view_ref());
    if (!status.is_ok()) return status;
    Tensor attn_output(tokens, model_dim);
    status = attention_.try_forward(
        attn_input.view_ref(), attn_output.mutable_view_ref());
    if (!status.is_ok()) return status;
    Tensor residual_1(tokens, model_dim);
    status = try_add_views(
        input, attn_output.view_ref(), residual_1.mutable_view_ref());
    if (!status.is_ok()) return status;

    Tensor ffn_input(tokens, model_dim);
    status = pre_ffn_norm_.try_forward(
        residual_1.view_ref(), ffn_input.mutable_view_ref());
    if (!status.is_ok()) return status;
    Tensor ffn_output(tokens, model_dim);
    status = feed_forward_.try_forward(
        ffn_input.view_ref(), ffn_output.mutable_view_ref());
    if (!status.is_ok()) return status;

    return try_add_views(residual_1.view_ref(), ffn_output.view_ref(), output);
}

cpp_llm_engine::core::Status DecoderLayer::try_prefill(
    const cpp_llm_engine::core::TensorViewRef& input,
    DecoderLayerContext* context,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    return try_forward_with_context(input, context, true, output);
}

cpp_llm_engine::core::Status DecoderLayer::try_decode(
    const cpp_llm_engine::core::TensorViewRef& input,
    DecoderLayerContext* context,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    return try_forward_with_context(input, context, false, output);
}

cpp_llm_engine::core::Status DecoderLayer::try_forward_with_context(
    const cpp_llm_engine::core::TensorViewRef& input,
    DecoderLayerContext* context,
    bool start_new_sequence,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    using cpp_llm_engine::core::Status;

    if (!ready()) return Status::error("decoder layer is not ready");
    if (context == nullptr || !context->ready()) {
        return Status::error("decoder layer requires a ready request context");
    }
    const auto input_status = validate_cpu_float32_matrix_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;

    const auto tokens = input.contract.shape.dimensions[0];
    const auto model_dim = attention_.model_dim();
    if (input.contract.shape.dimensions[1] != model_dim) {
        return Status::error("decoder layer input shape mismatch");
    }
    if (output.contract.shape.dimensions !=
        std::vector<std::size_t>{tokens, model_dim}) {
        return Status::error("decoder layer output shape mismatch");
    }
    if (context->kv_cache_.heads() != attention_.num_heads() ||
        context->kv_cache_.head_dim() != model_dim / attention_.num_heads()) {
        return Status::error("decoder layer request context shape mismatch");
    }
    if (!start_new_sequence && tokens != 1) {
        return Status::error("decoder step requires exactly one token");
    }
    if (!start_new_sequence && context->token_count() == 0) {
        return Status::error("decoder step requires a completed prefill");
    }

    DecoderLayerContext candidate = *context;
    if (start_new_sequence) candidate.reset();
    const std::size_t query_position_offset = candidate.token_count();

    Tensor attn_input(tokens, model_dim);
    auto status = pre_attn_norm_.try_forward(
        input, attn_input.mutable_view_ref());
    if (!status.is_ok()) return status;
    Tensor attn_output(tokens, model_dim);
    status = attention_.try_forward_with_cache(
        attn_input.view_ref(), query_position_offset, &candidate.kv_cache_,
        attn_output.mutable_view_ref());
    if (!status.is_ok()) return status;
    Tensor residual_1(tokens, model_dim);
    status = try_add_views(
        input, attn_output.view_ref(), residual_1.mutable_view_ref());
    if (!status.is_ok()) return status;

    Tensor ffn_input(tokens, model_dim);
    status = pre_ffn_norm_.try_forward(
        residual_1.view_ref(), ffn_input.mutable_view_ref());
    if (!status.is_ok()) return status;
    Tensor ffn_output(tokens, model_dim);
    status = feed_forward_.try_forward(
        ffn_input.view_ref(), ffn_output.mutable_view_ref());
    if (!status.is_ok()) return status;
    Tensor result(tokens, model_dim);
    status = try_add_views(
        residual_1.view_ref(), ffn_output.view_ref(), result.mutable_view_ref());
    if (!status.is_ok()) return status;

    *context = std::move(candidate);
    auto* output_data = static_cast<float*>(output.storage_data) +
                        output.contract.storage_offset_elements;
    std::copy(result.data.begin(), result.data.end(), output_data);
    return Status::ok();
}

bool DecoderLayer::ready() const {
    return pre_attn_norm_.ready() && attention_.ready() && pre_ffn_norm_.ready() && feed_forward_.ready();
}

std::size_t DecoderLayer::layer_index() const {
    return layer_index_;
}

cpp_llm_engine::core::DecoderLayerContract DecoderLayer::contract() const {
    const std::size_t model_dim = attention_.model_dim();
    const std::size_t head_count = attention_.num_heads();
    return {
        model_dim,
        feed_forward_.hidden_dim(),
        head_count,
        head_count,
        head_count == 0 ? 0 : model_dim / head_count,
        0,
        cpp_llm_engine::core::DecoderNormalizationKind::LayerNorm,
        cpp_llm_engine::core::DecoderActivationKind::Relu,
        cpp_llm_engine::core::DecoderPositionEncodingKind::None,
        true,
        true,
    };
}

}  // namespace chapter04_attention
