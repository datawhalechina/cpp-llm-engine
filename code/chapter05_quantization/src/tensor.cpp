// Chapter 05：实现量化教学 Tensor 的存储、形状与 dtype 访问。
// 边界：对象只服务本章 INT8/F32 示例和单元测试。
#include "chapter05_quantization/tensor.h"

#include <stdexcept>

namespace chapter05_quantization {

Tensor::Tensor() = default;

Tensor::Tensor(std::size_t rows,
               std::size_t cols,
               DType dtype,
               float scale,
               std::shared_ptr<std::vector<float>> float_storage,
               std::shared_ptr<std::vector<std::int8_t>> int8_storage)
    : rows_(rows),
      cols_(cols),
      dtype_(dtype),
      scale_(scale),
      float_storage_(std::move(float_storage)),
      int8_storage_(std::move(int8_storage)) {}

Tensor Tensor::from_float(std::size_t rows, std::size_t cols, const std::vector<float>& values) {
    if (values.size() != rows * cols) {
        return {};
    }
    return Tensor(rows, cols, DType::Float32, 1.0f, std::make_shared<std::vector<float>>(values), {});
}

Tensor Tensor::from_int8(std::size_t rows, std::size_t cols, const std::vector<std::int8_t>& values, float scale) {
    if (values.size() != rows * cols || scale <= 0.0f) {
        return {};
    }
    return Tensor(rows, cols, DType::Int8, scale, {}, std::make_shared<std::vector<std::int8_t>>(values));
}

bool Tensor::ready() const {
    if (rows_ == 0 || cols_ == 0) {
        return false;
    }
    if (dtype_ == DType::Float32) {
        return float_storage_ && float_storage_->size() == rows_ * cols_;
    }
    return int8_storage_ && int8_storage_->size() == rows_ * cols_ && scale_ > 0.0f;
}

bool Tensor::same_shape(const Tensor& other) const {
    return rows_ == other.rows_ && cols_ == other.cols_;
}

bool Tensor::is_float() const {
    return dtype_ == DType::Float32;
}

bool Tensor::is_int8() const {
    return dtype_ == DType::Int8;
}

std::size_t Tensor::rows() const {
    return rows_;
}

std::size_t Tensor::cols() const {
    return cols_;
}

DType Tensor::dtype() const {
    return dtype_;
}

float Tensor::scale() const {
    return scale_;
}

float Tensor::at(std::size_t row, std::size_t col) const {
    if (!ready() || row >= rows_ || col >= cols_) {
        throw std::out_of_range("tensor index out of range");
    }
    const std::size_t index = row * cols_ + col;
    if (dtype_ == DType::Float32) {
        return float_storage_->at(index);
    }
    return static_cast<float>(int8_storage_->at(index)) * scale_;
}

std::int8_t Tensor::at_int8(std::size_t row, std::size_t col) const {
    if (!ready() || dtype_ != DType::Int8 || row >= rows_ || col >= cols_) {
        throw std::out_of_range("tensor index out of range");
    }
    return int8_storage_->at(row * cols_ + col);
}

Tensor Tensor::dequantize() const {
    if (!ready()) {
        return {};
    }
    if (dtype_ == DType::Float32) {
        return *this;
    }

    std::vector<float> values;
    values.reserve(rows_ * cols_);
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            values.push_back(at(row, col));
        }
    }
    return Tensor::from_float(rows_, cols_, values);
}

}  // namespace chapter05_quantization
