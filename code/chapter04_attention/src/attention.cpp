// Chapter 04：MultiHeadAttention 实现
// 职责：执行 Q/K/V projection、Attention backend 调用和 output projection。
// 契约：输入 layout 为 [tokens, model_dim]，backend 看到连续 row-major buffer。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/attention.h"
#include "chapter04_attention/attention_backend.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace chapter04_attention {

MultiHeadAttention::MultiHeadAttention(std::size_t model_dim, std::size_t num_heads)
    : model_dim_(model_dim),
      num_heads_(num_heads),
      q_proj_(model_dim, model_dim),
      k_proj_(model_dim, model_dim),
      v_proj_(model_dim, model_dim),
      out_proj_(model_dim, model_dim),
      backend_(&cpu_reference_attention_backend()) {}

MultiHeadAttention::MultiHeadAttention(
    std::size_t model_dim,
    std::size_t num_heads,
    const cpp_llm_engine::core::AttentionBackend& backend)
    : model_dim_(model_dim),
      num_heads_(num_heads),
      q_proj_(model_dim, model_dim),
      k_proj_(model_dim, model_dim),
      v_proj_(model_dim, model_dim),
      out_proj_(model_dim, model_dim),
      backend_(&backend) {}

Tensor MultiHeadAttention::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != model_dim_) {
        return {};
    }

    Tensor output(input.rows, model_dim_);
    const auto status = try_forward(input.view_ref(), output.mutable_view_ref());
    if (!status.is_ok()) return {};
    return output;
}

cpp_llm_engine::core::Status MultiHeadAttention::try_forward(
    const cpp_llm_engine::core::TensorViewRef& input,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    using cpp_llm_engine::core::Status;

    if (!ready()) return Status::error("multi-head attention is not ready");
    const auto input_status = validate_cpu_float32_matrix_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;
    const auto tokens = input.contract.shape.dimensions[0];
    if (input.contract.shape.dimensions[1] != model_dim_) {
        return Status::error("multi-head attention input shape mismatch");
    }
    if (output.contract.shape.dimensions !=
        std::vector<std::size_t>{tokens, model_dim_}) {
        return Status::error("multi-head attention output shape mismatch");
    }

    const std::size_t head_dim = model_dim_ / num_heads_;
    Tensor q(tokens, model_dim_);
    Tensor k(tokens, model_dim_);
    Tensor v(tokens, model_dim_);
    auto status = q_proj_.try_forward(input, q.mutable_view_ref());
    if (!status.is_ok()) return status;
    status = k_proj_.try_forward(input, k.mutable_view_ref());
    if (!status.is_ok()) return status;
    status = v_proj_.try_forward(input, v.mutable_view_ref());
    if (!status.is_ok()) return status;

    Tensor combined(tokens, model_dim_);
    const cpp_llm_engine::core::AttentionRequest request{
        q.data.data(), k.data.data(), v.data.data(), combined.data.data(),
        tokens, tokens, num_heads_, head_dim, 0, true, 0.0f};
    status = backend_->compute(request);
    if (!status.is_ok()) return status;
    return out_proj_.try_forward(combined.view_ref(), output);
}

cpp_llm_engine::core::Status MultiHeadAttention::try_forward_with_cache(
    const cpp_llm_engine::core::TensorViewRef& input,
    std::size_t query_position_offset,
    KvCache* cache,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    using cpp_llm_engine::core::Status;

    if (!ready()) return Status::error("multi-head attention is not ready");
    if (cache == nullptr || !cache->ready()) {
        return Status::error("attention requires a ready KV cache");
    }
    const std::size_t head_dim = model_dim_ / num_heads_;
    if (cache->heads() != num_heads_ || cache->head_dim() != head_dim) {
        return Status::error("attention KV cache shape mismatch");
    }
    if (query_position_offset != cache->size()) {
        return Status::error("attention query position must match KV cache size");
    }

    const auto input_status = validate_cpu_float32_matrix_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;
    const auto tokens = input.contract.shape.dimensions[0];
    if (input.contract.shape.dimensions[1] != model_dim_) {
        return Status::error("multi-head attention input shape mismatch");
    }
    if (output.contract.shape.dimensions !=
        std::vector<std::size_t>{tokens, model_dim_}) {
        return Status::error("multi-head attention output shape mismatch");
    }

    Tensor q(tokens, model_dim_);
    Tensor k(tokens, model_dim_);
    Tensor v(tokens, model_dim_);
    auto status = q_proj_.try_forward(input, q.mutable_view_ref());
    if (!status.is_ok()) return status;
    status = k_proj_.try_forward(input, k.mutable_view_ref());
    if (!status.is_ok()) return status;
    status = v_proj_.try_forward(input, v.mutable_view_ref());
    if (!status.is_ok()) return status;

    KvCache candidate = *cache;
    if (!candidate.append(k.data.data(), v.data.data(), tokens)) {
        return Status::error("attention KV cache capacity exceeded");
    }

    Tensor combined(tokens, model_dim_);
    const cpp_llm_engine::core::AttentionRequest request{
        q.data.data(), candidate.key_data(), candidate.value_data(),
        combined.data.data(), tokens, candidate.size(), num_heads_, head_dim,
        query_position_offset, true, 0.0f};
    status = backend_->compute(request);
    if (!status.is_ok()) return status;

    Tensor projected(tokens, model_dim_);
    status = out_proj_.try_forward(combined.view_ref(), projected.mutable_view_ref());
    if (!status.is_ok()) return status;

    *cache = std::move(candidate);
    auto* output_data = static_cast<float*>(output.storage_data) +
                        output.contract.storage_offset_elements;
    std::copy(projected.data.begin(), projected.data.end(), output_data);
    return Status::ok();
}

bool MultiHeadAttention::ready() const {
    return model_dim_ > 0 && num_heads_ > 0 && model_dim_ % num_heads_ == 0 &&
           q_proj_.ready() && k_proj_.ready() && v_proj_.ready() && out_proj_.ready();
}

std::size_t MultiHeadAttention::model_dim() const {
    return model_dim_;
}

std::size_t MultiHeadAttention::num_heads() const {
    return num_heads_;
}

cpp_llm_engine::core::BackendKind MultiHeadAttention::backend_kind() const {
    return backend_->kind();
}

}  // namespace chapter04_attention
