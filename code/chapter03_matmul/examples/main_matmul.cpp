// Chapter 03：CPU 矩阵乘示例
// 场景：构造两个行主序 Matrix，通过 MatMulOp 计算二维输出。
// 观察：输出形状为 2x2，左上角为 58，右下角为 154。
// 运行：bash code/chapter03_matmul/scripts/run.sh
#include "chapter03_matmul/matmul_op.h"

#include <iostream>

int main() {
    chapter03_matmul::Matrix lhs(2, 3);
    lhs.at(0, 0) = 1.0f;
    lhs.at(0, 1) = 2.0f;
    lhs.at(0, 2) = 3.0f;
    lhs.at(1, 0) = 4.0f;
    lhs.at(1, 1) = 5.0f;
    lhs.at(1, 2) = 6.0f;

    chapter03_matmul::Matrix rhs(3, 2);
    rhs.at(0, 0) = 7.0f;
    rhs.at(0, 1) = 8.0f;
    rhs.at(1, 0) = 9.0f;
    rhs.at(1, 1) = 10.0f;
    rhs.at(2, 0) = 11.0f;
    rhs.at(2, 1) = 12.0f;

    chapter03_matmul::MatMulOp op;
    const chapter03_matmul::Matrix out = op.compute(lhs, rhs);

    std::cout << "[chapter03] matmul demo\n";
    std::cout << "[chapter03] output shape: " << out.rows << "x" << out.cols << "\n";
    std::cout << "[chapter03] value(0,0): " << out.at(0, 0) << "\n";
    std::cout << "[chapter03] value(1,1): " << out.at(1, 1) << "\n";
    return 0;
}
