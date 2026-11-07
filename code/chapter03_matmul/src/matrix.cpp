// Chapter 03：教学 Matrix 实现
// 数据：分配行主序元素存储，并把二维坐标转换为一维索引。
// 校验：构造时检查元素数溢出，访问时拒绝越界坐标。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#include "chapter03_matmul/matrix.h"

#include <limits>
#include <stdexcept>

namespace chapter03_matmul {

namespace {

bool checked_product(std::size_t lhs, std::size_t rhs, std::size_t& result) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

std::size_t checked_index(const Matrix& matrix, std::size_t row, std::size_t col) {
    if (row >= matrix.rows || col >= matrix.cols) {
        throw std::out_of_range("matrix index out of range");
    }
    std::size_t row_offset = 0;
    if (!checked_product(row, matrix.cols, row_offset) ||
        row_offset > std::numeric_limits<std::size_t>::max() - col) {
        throw std::out_of_range("matrix index overflows host size");
    }
    return row_offset + col;
}

}  // namespace

Matrix::Matrix(std::size_t r, std::size_t c, float value)
    : rows(r), cols(c) {
    std::size_t element_count = 0;
    if (!checked_product(r, c, element_count)) {
        throw std::length_error("matrix shape is too large");
    }
    data.assign(element_count, value);
}

float& Matrix::at(std::size_t row, std::size_t col) {
    return data.at(checked_index(*this, row, col));
}

float Matrix::at(std::size_t row, std::size_t col) const {
    return data.at(checked_index(*this, row, col));
}

bool Matrix::ready() const {
    std::size_t element_count = 0;
    const auto metadata = contract();
    return metadata.shape.checked_numel(&element_count).is_ok() &&
           element_count > 0 && data.size() == element_count &&
           metadata.validate_element_view(sizeof(float)).is_ok();
}

cpp_llm_engine::core::TensorContract Matrix::contract() const {
    return cpp_llm_engine::core::TensorContract{
        cpp_llm_engine::core::Shape{{rows, cols}},
        {cols, 1},
        cpp_llm_engine::core::DType::Float32,
        cpp_llm_engine::core::DeviceKind::CPU,
        0,
        data.size() * sizeof(float),
    };
}

}  // namespace chapter03_matmul
