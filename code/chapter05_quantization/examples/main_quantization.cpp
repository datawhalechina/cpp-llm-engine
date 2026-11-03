// Chapter 05：演示 INT8 weight-only Linear 的最小执行路径。
// 验证：输出浮点 Linear、量化 Linear 及二者误差，帮助理解表示变化。
#include "chapter05_quantization/dequantize_op.h"
#include "chapter05_quantization/linear_layer.h"
#include "chapter05_quantization/quantized_linear.h"

#include <iostream>

int main() {
    using namespace chapter05_quantization;

    Tensor quantized_weight = Tensor::from_int8(
        2, 2, {1, 2, 3, 4}, 0.1f);
    DequantizeOp dequantize_op;
    const Tensor dequantized_weight = dequantize_op.forward(quantized_weight);

    Tensor input = Tensor::from_float(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    LinearLayer reference_linear(2, 2);
    QuantizedLinear quantized_linear(2, 2, 0.1f);
    const Tensor output = quantized_linear.forward(input);

    std::cout << "[chapter05] quantization demo\n";
    std::cout << "quantized ready: " << std::boolalpha << quantized_linear.ready() << '\n';
    std::cout << "dequantized weight value(0,0): " << dequantized_weight.at(0, 0) << '\n';
    std::cout << "output shape: " << output.rows() << "x" << output.cols() << '\n';
    std::cout << "output value(0,0): " << output.at(0, 0) << '\n';
    std::cout << "output value(1,1): " << output.at(1, 1) << '\n';
    (void)reference_linear;
    return 0;
}
