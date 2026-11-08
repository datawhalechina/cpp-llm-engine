// Chapter 05：声明量化教学示例使用的最小二维 Tensor。
// 类型覆盖 F32 与 INT8，用于观察缩放因子和反量化结果。
// 验证入口：chapter05_quantization_test。
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace chapter05_quantization {

enum class DType {
    Float32,
    Int8,
};

class Tensor final {
public:
    Tensor();

    static Tensor from_float(std::size_t rows, std::size_t cols, const std::vector<float>& values);
    static Tensor from_int8(std::size_t rows, std::size_t cols, const std::vector<std::int8_t>& values, float scale);

    bool ready() const;
    bool same_shape(const Tensor& other) const;
    bool is_float() const;
    bool is_int8() const;

    std::size_t rows() const;
    std::size_t cols() const;
    DType dtype() const;
    float scale() const;

    float at(std::size_t row, std::size_t col) const;
    std::int8_t at_int8(std::size_t row, std::size_t col) const;

    Tensor dequantize() const;

private:
    Tensor(std::size_t rows,
           std::size_t cols,
           DType dtype,
           float scale,
           std::shared_ptr<std::vector<float>> float_storage,
           std::shared_ptr<std::vector<std::int8_t>> int8_storage);

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    DType dtype_ = DType::Float32;
    float scale_ = 1.0f;
    std::shared_ptr<std::vector<float>> float_storage_;
    std::shared_ptr<std::vector<std::int8_t>> int8_storage_;
};

}  // namespace chapter05_quantization
