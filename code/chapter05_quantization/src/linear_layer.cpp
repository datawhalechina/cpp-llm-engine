// Chapter 05：实现教学浮点 Linear 及其确定性权重和 bias。
// 用途：作为 QuantizedLinear 的小规模 correctness 对照。
#include "chapter05_quantization/linear_layer.h"

namespace chapter05_quantization {

namespace {

Tensor make_weight(std::size_t in_features, std::size_t out_features) {
    std::vector<float> values;
    values.reserve(in_features * out_features);
    for (std::size_t row = 0; row < in_features; ++row) {
        for (std::size_t col = 0; col < out_features; ++col) {
            values.push_back(0.1f * static_cast<float>((row + 1) * (col + 1)));
        }
    }
    return Tensor::from_float(in_features, out_features, values);
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.ready() || !rhs.ready() || lhs.cols() != rhs.rows()) {
        return {};
    }

    std::vector<float> values(lhs.rows() * rhs.cols(), 0.0f);
    for (std::size_t row = 0; row < lhs.rows(); ++row) {
        for (std::size_t col = 0; col < rhs.cols(); ++col) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < lhs.cols(); ++k) {
                sum += lhs.at(row, k) * rhs.at(k, col);
            }
            values[row * rhs.cols() + col] = sum;
        }
    }
    return Tensor::from_float(lhs.rows(), rhs.cols(), values);
}

Tensor add_bias(const Tensor& tensor, const std::vector<float>& bias) {
    if (!tensor.ready() || tensor.cols() != bias.size()) {
        return {};
    }

    std::vector<float> values;
    values.reserve(tensor.rows() * tensor.cols());
    for (std::size_t row = 0; row < tensor.rows(); ++row) {
        for (std::size_t col = 0; col < tensor.cols(); ++col) {
            values.push_back(tensor.at(row, col) + bias[col]);
        }
    }
    return Tensor::from_float(tensor.rows(), tensor.cols(), values);
}

}  // namespace

LinearLayer::LinearLayer(std::size_t in_features, std::size_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weight_(make_weight(in_features, out_features)),
      bias_(out_features, 0.0f) {}

Tensor LinearLayer::forward(const Tensor& input) const {
    if (!ready() || !input.ready() || input.cols() != in_features_) {
        return {};
    }
    return add_bias(matmul(input, weight_), bias_);
}

bool LinearLayer::ready() const {
    return in_features_ > 0 && out_features_ > 0 && weight_.ready() && bias_.size() == out_features_;
}

}  // namespace chapter05_quantization
