// Chapter 03：教学 Matrix 对象
// 数据：data 按行主序保存 Float32 元素，rows 和 cols 描述二维形状。
// 限制：元素数必须等于 rows * cols；尺寸溢出或索引越界时失败。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#pragma once

#include "cpp_llm_engine/core/tensor_contract.h"

#include <cstddef>
#include <vector>

namespace chapter03_matmul {

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, float value = 0.0f);

    float& at(std::size_t row, std::size_t col);
    float at(std::size_t row, std::size_t col) const;
    bool ready() const;
    cpp_llm_engine::core::TensorContract contract() const;
};

}  // namespace chapter03_matmul
