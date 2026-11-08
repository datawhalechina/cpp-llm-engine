// Chapter 02：CPU Float32 Tensor 及其 View 实现
// shape、stride 和 offset 决定逻辑索引对应的 Storage 位置。
// 地址计算检查溢出；try_* 原地失败不修改源对象。
// 验证：code/chapter02_tensor_ops/scripts/run.sh。
#include "chapter02_tensor_ops/tensor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace chapter02_tensor_ops {

namespace {

using cpp_llm_engine::core::Status;

bool checked_multiply(std::size_t lhs, std::size_t rhs, std::size_t& result) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) return false;
    result = lhs * rhs;
    return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t& result) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) return false;
    result = lhs + rhs;
    return true;
}

Status validate_shape(const std::vector<std::size_t>& shape, std::size_t* element_count) {
    if (element_count == nullptr) {
        return Status::error("tensor element-count output must not be null");
    }
    if (shape.empty() || shape.size() > 4) {
        return Status::error("tensor rank must be between 1 and 4");
    }
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension == 0) return Status::error("tensor dimensions must be non-zero");
        if (!checked_multiply(count, dimension, count)) {
            return Status::error("tensor shape is too large");
        }
    }
    *element_count = count;
    return Status::ok();
}

Status make_contiguous_strides(
    const std::vector<std::size_t>& shape,
    std::vector<std::size_t>* strides) {
    if (strides == nullptr) return Status::error("tensor strides output must not be null");
    std::size_t element_count = 0;
    const auto shape_status = validate_shape(shape, &element_count);
    if (!shape_status.is_ok()) return shape_status;

    strides->assign(shape.size(), 1);
    for (std::size_t axis = shape.size(); axis-- > 1;) {
        if (!checked_multiply((*strides)[axis], shape[axis], (*strides)[axis - 1])) {
            return Status::error("tensor stride overflows host size");
        }
    }
    return Status::ok();
}

Status checked_storage_index(
    std::size_t offset,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    const std::vector<std::size_t>& indices,
    std::size_t* result) {
    if (result == nullptr) return Status::error("tensor index output must not be null");
    if (shape.size() != strides.size() || indices.size() != shape.size()) {
        return Status::error("tensor index rank mismatch");
    }
    std::size_t index = offset;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (indices[axis] >= shape[axis]) return Status::error("tensor index out of range");
        std::size_t axis_offset = 0;
        if (!checked_multiply(indices[axis], strides[axis], axis_offset) ||
            !checked_add(index, axis_offset, index)) {
            return Status::error("tensor index overflow");
        }
    }
    *result = index;
    return Status::ok();
}

std::vector<std::size_t> indices_from_linear(
    std::size_t linear, const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> indices(shape.size(), 0);
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        indices[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    return indices;
}

}  // namespace

Tensor::Tensor() = default;

Tensor::Tensor(std::size_t rows, std::size_t cols, float value)
    : Tensor(std::vector<std::size_t>{rows, cols}, value) {}

Tensor::Tensor(std::vector<std::size_t> shape, float value)
    : shape_(std::move(shape)) {
    std::size_t element_count = 0;
    const auto shape_status = validate_shape(shape_, &element_count);
    if (!shape_status.is_ok()) throw std::length_error(shape_status.message());
    const auto stride_status = make_contiguous_strides(shape_, &strides_);
    if (!stride_status.is_ok()) throw std::length_error(stride_status.message());
    storage_ = std::make_shared<std::vector<float>>(element_count, value);
}

Tensor::Tensor(std::shared_ptr<std::vector<float>> storage,
               std::vector<std::size_t> shape,
               std::vector<std::size_t> strides,
               std::size_t offset)
    : storage_(std::move(storage)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset) {}

float& Tensor::at(std::size_t row, std::size_t col) {
    if (rank() != 2) throw std::out_of_range("two-dimensional tensor required");
    return at(std::vector<std::size_t>{row, col});
}

float Tensor::at(std::size_t row, std::size_t col) const {
    if (rank() != 2) throw std::out_of_range("two-dimensional tensor required");
    return at(std::vector<std::size_t>{row, col});
}

float& Tensor::at(const std::vector<std::size_t>& indices) {
    if (!ready()) throw std::out_of_range("tensor is not ready");
    std::size_t index = 0;
    const auto status = checked_storage_index(offset_, shape_, strides_, indices, &index);
    if (!status.is_ok()) throw std::out_of_range(status.message());
    return storage_->at(index);
}

float Tensor::at(const std::vector<std::size_t>& indices) const {
    if (!ready()) throw std::out_of_range("tensor is not ready");
    std::size_t index = 0;
    const auto status = checked_storage_index(offset_, shape_, strides_, indices, &index);
    if (!status.is_ok()) throw std::out_of_range(status.message());
    return storage_->at(index);
}

float* Tensor::data() {
    return ready() && is_contiguous() ? storage_->data() + offset_ : nullptr;
}

const float* Tensor::data() const {
    return ready() && is_contiguous() ? storage_->data() + offset_ : nullptr;
}

bool Tensor::ready() const {
    return storage_ && contract().validate_element_view(sizeof(float)).is_ok();
}

bool Tensor::is_contiguous() const {
    if (!ready()) return false;
    std::size_t expected_stride = 1;
    for (std::size_t axis = rank(); axis-- > 0;) {
        if (shape_[axis] != 1 && strides_[axis] != expected_stride) return false;
        if (!checked_multiply(expected_stride, shape_[axis], expected_stride)) return false;
    }
    return true;
}

bool Tensor::shares_storage_with(const Tensor& other) const {
    return storage_ && other.storage_ && storage_ == other.storage_;
}

std::size_t Tensor::storage_use_count() const {
    return storage_ ? storage_.use_count() : 0;
}

Tensor Tensor::contiguous() const {
    Tensor output;
    if (!try_contiguous(&output).is_ok()) return {};
    return output;
}

Status Tensor::try_contiguous(Tensor* output) const {
    if (output == nullptr) return Status::error("contiguous output must not be null");
    if (output == this) {
        Tensor result;
        const auto status = try_contiguous(&result);
        if (status.is_ok()) *output = std::move(result);
        return status;
    }
    if (!ready()) {
        *output = {};
        return Status::error("contiguous source tensor is not ready");
    }
    if (is_contiguous()) {
        *output = *this;
        return Status::ok();
    }

    Tensor materialized(shape_);
    for (std::size_t linear = 0; linear < numel(); ++linear) {
        const auto indices = indices_from_linear(linear, shape_);
        materialized.at(indices) = at(indices);
    }
    *output = std::move(materialized);
    return Status::ok();
}

Tensor Tensor::reshape(std::size_t rows, std::size_t cols) const {
    return reshape(std::vector<std::size_t>{rows, cols});
}

Tensor Tensor::reshape(std::vector<std::size_t> shape) const {
    Tensor output;
    if (!try_reshape(std::move(shape), &output).is_ok()) return {};
    return output;
}

Status Tensor::try_reshape(std::size_t rows, std::size_t cols, Tensor* output) const {
    if (rows == 0 || cols == 0) {
        if (output != nullptr && output != this) *output = {};
        return Status::error("reshape dimensions must be non-zero");
    }
    return try_reshape(std::vector<std::size_t>{rows, cols}, output);
}

Status Tensor::try_reshape(std::vector<std::size_t> shape, Tensor* output) const {
    if (output == nullptr) return Status::error("reshape output must not be null");
    if (output == this) {
        Tensor result;
        const auto status = try_reshape(std::move(shape), &result);
        if (status.is_ok()) *output = std::move(result);
        return status;
    }
    *output = {};
    if (!ready()) return Status::error("reshape source tensor is not ready");
    if (!is_contiguous()) return Status::error("reshape requires a contiguous tensor");

    std::size_t new_element_count = 0;
    const auto shape_status = validate_shape(shape, &new_element_count);
    if (!shape_status.is_ok()) return shape_status;
    if (new_element_count != numel()) {
        return Status::error("reshape element count must not change");
    }
    std::vector<std::size_t> strides;
    const auto stride_status = make_contiguous_strides(shape, &strides);
    if (!stride_status.is_ok()) return stride_status;
    *output = Tensor(storage_, std::move(shape), std::move(strides), offset_);
    return Status::ok();
}

Tensor Tensor::slice(
    std::size_t row_start,
    std::size_t row_count,
    std::size_t col_start,
    std::size_t col_count) const {
    Tensor output;
    if (!try_slice(row_start, row_count, col_start, col_count, &output).is_ok()) return {};
    return output;
}

Tensor Tensor::slice(std::size_t axis, std::size_t start, std::size_t count) const {
    Tensor output;
    if (!try_slice(axis, start, count, &output).is_ok()) return {};
    return output;
}

Status Tensor::try_slice(
    std::size_t row_start,
    std::size_t row_count,
    std::size_t col_start,
    std::size_t col_count,
    Tensor* output) const {
    if (output == nullptr) return Status::error("slice output must not be null");
    if (output == this) {
        Tensor result;
        const auto status = try_slice(
            row_start, row_count, col_start, col_count, &result);
        if (status.is_ok()) *output = std::move(result);
        return status;
    }
    *output = {};
    if (!ready()) return Status::error("slice source tensor is not ready");
    if (rank() != 2) return Status::error("two-dimensional tensor required");
    if (row_start > shape_[0] || col_start > shape_[1]) {
        return Status::error("slice start is outside tensor shape");
    }
    if (row_count == 0 || col_count == 0) {
        return Status::error("slice dimensions must be non-zero");
    }
    if (row_count > shape_[0] - row_start || col_count > shape_[1] - col_start) {
        return Status::error("slice range exceeds tensor shape");
    }
    std::size_t row_offset = 0;
    std::size_t col_offset = 0;
    std::size_t new_offset = offset_;
    if (!checked_multiply(row_start, strides_[0], row_offset) ||
        !checked_multiply(col_start, strides_[1], col_offset) ||
        !checked_add(new_offset, row_offset, new_offset) ||
        !checked_add(new_offset, col_offset, new_offset)) {
        return Status::error("slice offset overflows host size");
    }
    auto shape = shape_;
    shape[0] = row_count;
    shape[1] = col_count;
    *output = Tensor(storage_, std::move(shape), strides_, new_offset);
    return Status::ok();
}

Status Tensor::try_slice(
    std::size_t axis,
    std::size_t start,
    std::size_t count,
    Tensor* output) const {
    if (output == nullptr) return Status::error("slice output must not be null");
    if (output == this) {
        Tensor result;
        const auto status = try_slice(axis, start, count, &result);
        if (status.is_ok()) *output = std::move(result);
        return status;
    }
    *output = {};
    if (!ready()) return Status::error("slice source tensor is not ready");
    if (axis >= rank()) return Status::error("slice axis is out of range");
    if (start > shape_[axis]) return Status::error("slice start is outside tensor shape");
    if (count == 0) return Status::error("slice dimensions must be non-zero");
    if (count > shape_[axis] - start) return Status::error("slice range exceeds tensor shape");

    std::size_t axis_offset = 0;
    std::size_t new_offset = offset_;
    if (!checked_multiply(start, strides_[axis], axis_offset) ||
        !checked_add(new_offset, axis_offset, new_offset)) {
        return Status::error("slice offset overflows host size");
    }
    auto shape = shape_;
    shape[axis] = count;
    *output = Tensor(storage_, std::move(shape), strides_, new_offset);
    return Status::ok();
}

Tensor Tensor::transpose() const {
    return transpose(0, 1);
}

Tensor Tensor::transpose(std::size_t axis_a, std::size_t axis_b) const {
    Tensor output;
    if (!try_transpose(axis_a, axis_b, &output).is_ok()) return {};
    return output;
}

Status Tensor::try_transpose(Tensor* output) const {
    if (output == this) {
        Tensor result;
        const auto status = try_transpose(&result);
        if (status.is_ok()) *output = std::move(result);
        return status;
    }
    if (rank() != 2) {
        if (output != nullptr) *output = {};
        return Status::error("two-dimensional tensor required");
    }
    return try_transpose(0, 1, output);
}

Status Tensor::try_transpose(
    std::size_t axis_a, std::size_t axis_b, Tensor* output) const {
    if (output == nullptr) return Status::error("transpose output must not be null");
    if (output == this) {
        Tensor result;
        const auto status = try_transpose(axis_a, axis_b, &result);
        if (status.is_ok()) *output = std::move(result);
        return status;
    }
    *output = {};
    if (!ready()) return Status::error("transpose source tensor is not ready");
    if (axis_a >= rank() || axis_b >= rank()) {
        return Status::error("transpose axis is out of range");
    }
    auto shape = shape_;
    auto strides = strides_;
    std::swap(shape[axis_a], shape[axis_b]);
    std::swap(strides[axis_a], strides[axis_b]);
    *output = Tensor(storage_, std::move(shape), std::move(strides), offset_);
    return Status::ok();
}

cpp_llm_engine::core::TensorContract Tensor::contract() const {
    return cpp_llm_engine::core::TensorContract{
        cpp_llm_engine::core::Shape{shape_},
        strides_,
        cpp_llm_engine::core::DType::Float32,
        cpp_llm_engine::core::DeviceKind::CPU,
        offset_,
        storage_ ? storage_->size() * sizeof(float) : 0,
    };
}

cpp_llm_engine::core::TensorViewRef Tensor::view_ref() const {
    return cpp_llm_engine::core::TensorViewRef{
        storage_ ? static_cast<const void*>(storage_->data()) : nullptr,
        contract(),
    };
}

cpp_llm_engine::core::MutableTensorViewRef Tensor::mutable_view_ref() {
    return cpp_llm_engine::core::MutableTensorViewRef{
        storage_ ? static_cast<void*>(storage_->data()) : nullptr,
        contract(),
    };
}

std::size_t Tensor::rows() const {
    if (rank() != 2) throw std::out_of_range("two-dimensional tensor required");
    return shape_[0];
}

std::size_t Tensor::cols() const {
    if (rank() != 2) throw std::out_of_range("two-dimensional tensor required");
    return shape_[1];
}

std::size_t Tensor::rank() const {
    return shape_.size();
}

std::size_t Tensor::numel() const {
    std::size_t result = 0;
    return validate_shape(shape_, &result).is_ok() ? result : 0;
}

const std::vector<std::size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<std::size_t>& Tensor::strides() const {
    return strides_;
}

std::size_t Tensor::offset() const {
    return offset_;
}

std::size_t Tensor::row_stride() const {
    if (rank() != 2) throw std::out_of_range("two-dimensional tensor required");
    return strides_[0];
}

std::size_t Tensor::col_stride() const {
    if (rank() != 2) throw std::out_of_range("two-dimensional tensor required");
    return strides_[1];
}

}  // namespace chapter02_tensor_ops
