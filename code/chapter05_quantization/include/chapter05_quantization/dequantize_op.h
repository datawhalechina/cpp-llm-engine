// Chapter 05：声明教学 INT8 Tensor 的反量化算子。
// 契约：输入必须携带量化 scale，输出为同形状浮点 Tensor。
// 验证入口：chapter05_quantization_test。
#pragma once

#include "chapter05_quantization/tensor.h"

namespace chapter05_quantization {

class DequantizeOp final {
public:
    Tensor forward(const Tensor& input) const;
};

}  // namespace chapter05_quantization
