// Chapter 02：Tensor view 接口
// 职责：声明共享 storage 的 view 操作和显式连续化接口。
// 契约：view 操作不复制；try_* 支持原地输出且失败时保留源对象。
// 验证：chapter02_tensor_ops_test.cpp 和 chapter02_nd_tensor_test.cpp。
#pragma once

#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/core/tensor_view_ref.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace chapter02_tensor_ops {

class Tensor final {
public:
    Tensor();
    Tensor(std::size_t rows, std::size_t cols, float value = 0.0f);
    explicit Tensor(std::vector<std::size_t> shape, float value = 0.0f);

    float& at(std::size_t row, std::size_t col);
    float at(std::size_t row, std::size_t col) const;
    float& at(const std::vector<std::size_t>& indices);
    float at(const std::vector<std::size_t>& indices) const;
    float* data();
    const float* data() const;

    bool ready() const;
    bool is_contiguous() const;
    bool shares_storage_with(const Tensor& other) const;
    std::size_t storage_use_count() const;

    Tensor contiguous() const;
    Tensor reshape(std::size_t rows, std::size_t cols) const;
    Tensor reshape(std::vector<std::size_t> shape) const;
    Tensor slice(std::size_t row_start, std::size_t row_count, std::size_t col_start, std::size_t col_count) const;
    Tensor slice(std::size_t axis, std::size_t start, std::size_t count) const;
    Tensor transpose() const;
    Tensor transpose(std::size_t axis_a, std::size_t axis_b) const;
    cpp_llm_engine::core::Status try_contiguous(Tensor* output) const;
    cpp_llm_engine::core::Status try_reshape(
        std::size_t rows, std::size_t cols, Tensor* output) const;
    cpp_llm_engine::core::Status try_reshape(
        std::vector<std::size_t> shape, Tensor* output) const;
    cpp_llm_engine::core::Status try_slice(
        std::size_t row_start,
        std::size_t row_count,
        std::size_t col_start,
        std::size_t col_count,
        Tensor* output) const;
    cpp_llm_engine::core::Status try_slice(
        std::size_t axis, std::size_t start, std::size_t count, Tensor* output) const;
    cpp_llm_engine::core::Status try_transpose(Tensor* output) const;
    cpp_llm_engine::core::Status try_transpose(
        std::size_t axis_a, std::size_t axis_b, Tensor* output) const;
    cpp_llm_engine::core::TensorContract contract() const;
    cpp_llm_engine::core::TensorViewRef view_ref() const;
    cpp_llm_engine::core::MutableTensorViewRef mutable_view_ref();

    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t rank() const;
    std::size_t numel() const;
    const std::vector<std::size_t>& shape() const;
    const std::vector<std::size_t>& strides() const;
    std::size_t offset() const;
    std::size_t row_stride() const;
    std::size_t col_stride() const;

private:
    Tensor(std::shared_ptr<std::vector<float>> storage,
           std::vector<std::size_t> shape,
           std::vector<std::size_t> strides,
           std::size_t offset);

    std::shared_ptr<std::vector<float>> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t offset_ = 0;
};

}  // namespace chapter02_tensor_ops
