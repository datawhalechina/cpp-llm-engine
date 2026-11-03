// Chapter 04：FeedForward 实现
// 职责：执行 decoder FFN 的两次 Linear 和激活函数。
// 契约：保持输入 token 数和 model_dim，hidden_dim 只影响中间结果。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/feed_forward.h"

#include <algorithm>

namespace chapter04_attention {

FeedForward::FeedForward(std::size_t model_dim, std::size_t hidden_dim)
    : model_dim_(model_dim),
      hidden_dim_(hidden_dim),
      fc1_(model_dim, hidden_dim),
      fc2_(hidden_dim, model_dim) {}

Tensor FeedForward::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != model_dim_) {
        return {};
    }

    Tensor output(input.rows, model_dim_);
    const auto status = try_forward(input.view_ref(), output.mutable_view_ref());
    if (!status.is_ok()) return {};
    return output;
}

cpp_llm_engine::core::Status FeedForward::try_forward(
    const cpp_llm_engine::core::TensorViewRef& input,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    using cpp_llm_engine::core::Status;

    if (!ready()) return Status::error("feed forward is not ready");
    const auto input_status = validate_cpu_float32_matrix_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;
    const auto rows = input.contract.shape.dimensions[0];
    if (input.contract.shape.dimensions[1] != model_dim_) {
        return Status::error("feed forward input shape mismatch");
    }
    if (output.contract.shape.dimensions !=
        std::vector<std::size_t>{rows, model_dim_}) {
        return Status::error("feed forward output shape mismatch");
    }

    Tensor hidden(rows, hidden_dim_);
    auto status = fc1_.try_forward(input, hidden.mutable_view_ref());
    if (!status.is_ok()) return status;
    Tensor activated(rows, hidden_dim_);
    status = try_apply_activation(
        hidden.view_ref(), ActivationKind::Relu, activated.mutable_view_ref());
    if (!status.is_ok()) return status;
    return fc2_.try_forward(activated.view_ref(), output);
}

bool FeedForward::ready() const {
    return model_dim_ > 0 && hidden_dim_ > 0 && fc1_.ready() && fc2_.ready();
}

std::size_t FeedForward::model_dim() const {
    return model_dim_;
}

std::size_t FeedForward::hidden_dim() const {
    return hidden_dim_;
}

Tensor apply_activation(const Tensor& input, ActivationKind kind) {
    if (!input.ready()) {
        return {};
    }

    Tensor out(input.rows, input.cols);
    const auto status =
        try_apply_activation(input.view_ref(), kind, out.mutable_view_ref());
    if (!status.is_ok()) return {};
    return out;
}

cpp_llm_engine::core::Status try_apply_activation(
    const cpp_llm_engine::core::TensorViewRef& input,
    ActivationKind kind,
    const cpp_llm_engine::core::MutableTensorViewRef& output) {
    using cpp_llm_engine::core::Status;

    const auto input_status = validate_cpu_float32_matrix_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;
    if (input.contract.shape.dimensions != output.contract.shape.dimensions) {
        return Status::error("activation output shape mismatch");
    }

    std::size_t elements = 0;
    const auto shape_status = input.contract.shape.checked_numel(&elements);
    if (!shape_status.is_ok()) return shape_status;
    const auto* input_data = static_cast<const float*>(input.storage_data) +
                             input.contract.storage_offset_elements;
    std::vector<float> result(elements, 0.0f);
    for (std::size_t i = 0; i < elements; ++i) {
        switch (kind) {
            case ActivationKind::Relu:
            default:
                result[i] = input_data[i] > 0.0f ? input_data[i] : 0.0f;
                break;
        }
    }
    auto* output_data = static_cast<float*>(output.storage_data) +
                        output.contract.storage_offset_elements;
    std::copy(result.begin(), result.end(), output_data);
    return Status::ok();
}

}  // namespace chapter04_attention
