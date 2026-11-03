// Chapter 04：LayerNorm 实现
// 职责：通过公共 View 对每个 token 行计算均值、方差并应用 gamma/beta。
// 契约：只接受 CPU Float32 连续二维 View；失败不修改输出。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/layer_norm.h"

#include <algorithm>
#include <cmath>

namespace chapter04_attention {

LayerNorm::LayerNorm(std::size_t dim, float epsilon)
    : dim_(dim), epsilon_(epsilon), gamma_(dim, 1.0f), beta_(dim, 0.0f) {}

Tensor LayerNorm::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != dim_) {
        return {};
    }

    Tensor out(input.rows, input.cols);
    const auto status = try_forward(input.view_ref(), out.mutable_view_ref());
    if (!status.is_ok()) return {};
    return out;
}

cpp_llm_engine::core::Status LayerNorm::try_forward(
    const cpp_llm_engine::core::TensorViewRef& input,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    using cpp_llm_engine::core::Status;

    if (!ready()) return Status::error("layer norm is not ready");
    const auto input_status = validate_cpu_float32_matrix_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;
    const auto& input_shape = input.contract.shape.dimensions;
    if (input_shape[1] != dim_) {
        return Status::error("layer norm input shape mismatch");
    }
    if (output.contract.shape.dimensions != input_shape) {
        return Status::error("layer norm output shape mismatch");
    }

    const auto rows = input_shape[0];
    const auto cols = input_shape[1];
    const auto* input_data = static_cast<const float*>(input.storage_data) +
                             input.contract.storage_offset_elements;
    std::vector<float> result(rows * cols, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += input_data[row * cols + col];
        }
        const float mean = sum / static_cast<float>(cols);

        float var_sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            const float diff = input_data[row * cols + col] - mean;
            var_sum += diff * diff;
        }
        const float variance = var_sum / static_cast<float>(cols);
        const float denom = std::sqrt(variance + epsilon_);

        for (std::size_t col = 0; col < cols; ++col) {
            const float normalized =
                (input_data[row * cols + col] - mean) / denom;
            result[row * cols + col] = normalized * gamma_[col] + beta_[col];
        }
    }
    auto* output_data = static_cast<float*>(output.storage_data) +
                        output.contract.storage_offset_elements;
    std::copy(result.begin(), result.end(), output_data);
    return Status::ok();
}

bool LayerNorm::ready() const {
    return dim_ > 0 && gamma_.size() == dim_ && beta_.size() == dim_;
}

std::size_t LayerNorm::dim() const {
    return dim_;
}

float LayerNorm::epsilon() const {
    return epsilon_;
}

}  // namespace chapter04_attention
