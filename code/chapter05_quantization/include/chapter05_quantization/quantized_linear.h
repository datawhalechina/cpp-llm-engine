// Chapter 05：声明教学 INT8 weight-only Linear。
// 职责：保存量化权重，并通过反量化 baseline 完成前向计算。
// 验证入口：chapter05_quantization_test。
#pragma once

#include "chapter05_quantization/dequantize_op.h"
#include "chapter05_quantization/linear_layer.h"

#include <cstddef>

namespace chapter05_quantization {

class QuantizedLinear final {
public:
    QuantizedLinear(std::size_t in_features = 0,
                    std::size_t out_features = 0,
                    float weight_scale = 0.1f);

    Tensor forward(const Tensor& input) const;
    bool ready() const;

    const Tensor& quantized_weight() const;

private:
    std::size_t in_features_ = 0;
    std::size_t out_features_ = 0;
    float weight_scale_ = 1.0f;
    Tensor quantized_weight_;
    DequantizeOp dequantize_op_;
    std::vector<float> bias_;
};

}  // namespace chapter05_quantization
