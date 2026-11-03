// Chapter 02：Tensor view 示例
// 职责：演示二维 view 与三维 shape/stride 的可观察结果。
// 运行：bash code/chapter02_tensor_ops/scripts/run.sh。
// 验证：输出 shares storage=true，并打印 slice offset/stride。
#include "chapter02_tensor_ops/tensor.h"

#include <iostream>
#include <vector>

int main() {
    chapter02_tensor_ops::Tensor tensor(2, 3);
    tensor.at(0, 0) = 1.0f;
    tensor.at(0, 1) = 2.0f;
    tensor.at(0, 2) = 3.0f;
    tensor.at(1, 0) = 4.0f;
    tensor.at(1, 1) = 5.0f;
    tensor.at(1, 2) = 6.0f;

    const chapter02_tensor_ops::Tensor reshaped = tensor.reshape(3, 2);
    const chapter02_tensor_ops::Tensor sliced = tensor.slice(0, 2, 1, 2);
    const chapter02_tensor_ops::Tensor transposed = tensor.transpose();
    const chapter02_tensor_ops::Tensor qkv(std::vector<std::size_t>{2, 3, 4});

    std::cout << "[chapter02] tensor ops demo\n";
    std::cout << "tensor ready: " << std::boolalpha << tensor.ready() << '\n';
    std::cout << "reshaped shape: " << reshaped.rows() << "x" << reshaped.cols() << '\n';
    std::cout << "sliced shape: " << sliced.rows() << "x" << sliced.cols() << '\n';
    std::cout << "transposed shape/stride: " << transposed.rows() << "x" << transposed.cols()
              << "/" << transposed.row_stride() << "/" << transposed.col_stride() << '\n';
    std::cout << "3d qkv shape/stride: "
              << qkv.shape()[0] << "x" << qkv.shape()[1] << "x" << qkv.shape()[2] << "/"
              << qkv.strides()[0] << "/" << qkv.strides()[1] << "/" << qkv.strides()[2]
              << '\n';
    std::cout << "slice offset/stride: " << sliced.offset() << "/" << sliced.row_stride() << "/"
              << sliced.col_stride() << '\n';
    std::cout << "shares storage: " << std::boolalpha << tensor.shares_storage_with(reshaped) << '\n';
    std::cout << "slice value(0,0): " << sliced.at(0, 0) << '\n';
    return 0;
}
