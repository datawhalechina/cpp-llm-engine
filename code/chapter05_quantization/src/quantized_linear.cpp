// Chapter 05：实现教学 INT8 weight-only Linear。
// 路径：构造量化权重，反量化为 F32，再执行矩阵乘和 bias。
#include "chapter05_quantization/quantized_linear.h"

#include <vector>

namespace chapter05_quantization {

namespace {

Tensor make_quantized_weight(std::size_t in_features, std::size_t out_features, float scale) {
    std::vector<std::int8_t> values;
    values.reserve(in_features * out_features);
    for (std::size_t row = 0; row < in_features; ++row) {
        for (std::size_t col = 0; col < out_features; ++col) {
            const float real_value = 0.1f * static_cast<float>((row + 1) * (col + 1));
            const auto quantized = static_cast<std::int8_t>(real_value / scale);
            values.push_back(quantized);
        }
    }
    return Tensor::from_int8(in_features, out_features, values, scale);
}

}  // namespace

QuantizedLinear::QuantizedLinear(std::size_t in_features,
                                 std::size_t out_features,
                                 float weight_scale)
    : in_features_(in_features),
      out_features_(out_features),
      weight_scale_(weight_scale),
      quantized_weight_(make_quantized_weight(in_features, out_features, weight_scale)),
      bias_(out_features, 0.0f) {}

Tensor QuantizedLinear::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols() != in_features_) {
        return {};
    }
    const Tensor dequantized = dequantize_op_.forward(quantized_weight_);
    LinearLayer linear(in_features_, out_features_);

    // Use the dequantized weight as the effective path.
    // The tiny tutorial version keeps the implementation readable and testable.
    (void)linear;

    std::vector<float> values;
    values.reserve(input.rows() * out_features_);
    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t col = 0; col < out_features_; ++col) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < input.cols(); ++k) {
                sum += input.at(row, k) * dequantized.at(k, col);
            }
            values.push_back(sum + bias_[col]);
        }
    }
    return Tensor::from_float(input.rows(), out_features_, values);
}

bool QuantizedLinear::ready() const {
    return in_features_ > 0 && out_features_ > 0 &&
           weight_scale_ > 0.0f &&
           quantized_weight_.ready() &&
           quantized_weight_.is_int8() &&
           bias_.size() == out_features_;
}

const Tensor& QuantizedLinear::quantized_weight() const {
    return quantized_weight_;
}

}  // namespace chapter05_quantization
