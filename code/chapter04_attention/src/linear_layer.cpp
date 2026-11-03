// Chapter 04：Decoder Linear 实现
// 职责：保存教学 float 权重，并通过公共 View 执行 row-major Linear。
// 契约：只接受 CPU Float32 连续二维 View；失败不修改输出。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/linear_layer.h"

#include <algorithm>

namespace chapter04_attention {

namespace {

Tensor make_weight(std::size_t in_features, std::size_t out_features) {
    Tensor weight(in_features, out_features);
    for (std::size_t row = 0; row < in_features; ++row) {
        for (std::size_t col = 0; col < out_features; ++col) {
            weight.at(row, col) = 0.01f * static_cast<float>((row + 1) * (col + 1));
        }
    }
    return weight;
}

}  // namespace

LinearLayer::LinearLayer(std::size_t in_features, std::size_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(make_weight(in_features, out_features)),
      bias_(out_features, 0.0f) {}

Tensor LinearLayer::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols != in_features_) {
        return {};
    }

    Tensor out(input.rows, out_features_);
    const auto status = try_forward(input.view_ref(), out.mutable_view_ref());
    if (!status.is_ok()) return {};
    return out;
}

cpp_llm_engine::core::Status LinearLayer::try_forward(
    const cpp_llm_engine::core::TensorViewRef& input,
    const cpp_llm_engine::core::MutableTensorViewRef& output) const {
    using cpp_llm_engine::core::Status;

    if (!ready()) return Status::error("linear layer is not ready");
    const auto input_status = validate_cpu_float32_matrix_view(input);
    if (!input_status.is_ok()) return input_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;

    const auto rows = input.contract.shape.dimensions[0];
    if (input.contract.shape.dimensions[1] != in_features_) {
        return Status::error("linear input shape mismatch");
    }
    if (output.contract.shape.dimensions !=
        std::vector<std::size_t>{rows, out_features_}) {
        return Status::error("linear output shape mismatch");
    }

    const auto* input_data = static_cast<const float*>(input.storage_data) +
                             input.contract.storage_offset_elements;
    std::vector<float> result(rows * out_features_, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < out_features_; ++col) {
            float value = bias_[col];
            for (std::size_t inner = 0; inner < in_features_; ++inner) {
                value += input_data[row * in_features_ + inner] *
                         weight_.at(inner, col);
            }
            result[row * out_features_ + col] = value;
        }
    }
    auto* output_data = static_cast<float*>(output.storage_data) +
                        output.contract.storage_offset_elements;
    std::copy(result.begin(), result.end(), output_data);
    return Status::ok();
}

bool LinearLayer::ready() const {
    return in_features_ > 0 && out_features_ > 0 &&
           weight_.ready() && weight_.rows == in_features_ && weight_.cols == out_features_ &&
           bias_.size() == out_features_;
}

std::size_t LinearLayer::in_features() const {
    return in_features_;
}

std::size_t LinearLayer::out_features() const {
    return out_features_;
}

}  // namespace chapter04_attention
