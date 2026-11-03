// Chapter 04：Decoder 教学 Tensor 接口
// 职责：提供 row-major Tensor、矩阵操作、列切片和拼接辅助函数。
// 契约：算子失败返回空 Tensor，at() 对 shape 越界抛出异常。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/core/tensor_view_ref.h"

#include <cstddef>
#include <vector>

namespace chapter04_attention {

struct Tensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> data;

    Tensor() = default;
    Tensor(std::size_t r, std::size_t c, float value = 0.0f);

    float& at(std::size_t row, std::size_t col);
    float at(std::size_t row, std::size_t col) const;
    bool ready() const;
    cpp_llm_engine::core::TensorContract contract() const;
    cpp_llm_engine::core::TensorViewRef view_ref() const;
    cpp_llm_engine::core::MutableTensorViewRef mutable_view_ref();
};

cpp_llm_engine::core::Status validate_cpu_float32_matrix_view(
    const cpp_llm_engine::core::TensorViewRef& view);
cpp_llm_engine::core::Status try_add_views(
    const cpp_llm_engine::core::TensorViewRef& lhs,
    const cpp_llm_engine::core::TensorViewRef& rhs,
    const cpp_llm_engine::core::MutableTensorViewRef& output);
Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor transpose(const Tensor& tensor);
Tensor matmul(const Tensor& lhs, const Tensor& rhs);
Tensor slice_columns(const Tensor& tensor, std::size_t start_col, std::size_t width);
Tensor concatenate_columns(const std::vector<Tensor>& parts);
void scale_in_place(Tensor& tensor, float factor);

}  // namespace chapter04_attention
