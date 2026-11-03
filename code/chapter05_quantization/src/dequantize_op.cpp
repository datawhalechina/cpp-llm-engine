// Chapter 05：实现 INT8 Tensor 到 F32 Tensor 的 reference 反量化。
// 公式：每个元素使用 quantized_value * scale 恢复浮点近似值。
#include "chapter05_quantization/dequantize_op.h"

namespace chapter05_quantization {

Tensor DequantizeOp::forward(const Tensor& input) const {
    return input.dequantize();
}

}  // namespace chapter05_quantization
