// Chapter 04：Decoder Tensor 和辅助算子实现
// 职责：实现 add、transpose、matmul、列切片、拼接和缩放。
// 契约：操作失败返回空 Tensor，所有结果使用独立 row-major storage。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/tensor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace chapter04_attention {

Tensor::Tensor(std::size_t r, std::size_t c, float value) {
    if (r == 0 || c == 0 || r > std::numeric_limits<std::size_t>::max() / c) {
        return;
    }
    rows = r;
    cols = c;
    data.assign(rows * cols, value);
}

float& Tensor::at(std::size_t row, std::size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("tensor index out of range");
    }
    return data[row * cols + col];
}

float Tensor::at(std::size_t row, std::size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("tensor index out of range");
    }
    return data[row * cols + col];
}

bool Tensor::ready() const {
    return rows > 0 && cols > 0 &&
           rows <= std::numeric_limits<std::size_t>::max() / cols &&
           data.size() == rows * cols;
}

cpp_llm_engine::core::TensorContract Tensor::contract() const {
    const std::size_t storage_bytes =
        data.size() <= std::numeric_limits<std::size_t>::max() / sizeof(float)
            ? data.size() * sizeof(float)
            : 0;
    return cpp_llm_engine::core::TensorContract{
        cpp_llm_engine::core::Shape{{rows, cols}},
        {cols, 1},
        cpp_llm_engine::core::DType::Float32,
        cpp_llm_engine::core::DeviceKind::CPU,
        0,
        storage_bytes,
    };
}

cpp_llm_engine::core::TensorViewRef Tensor::view_ref() const {
    return cpp_llm_engine::core::TensorViewRef{data.data(), contract()};
}

cpp_llm_engine::core::MutableTensorViewRef Tensor::mutable_view_ref() {
    return cpp_llm_engine::core::MutableTensorViewRef{data.data(), contract()};
}

cpp_llm_engine::core::Status validate_cpu_float32_matrix_view(
    const cpp_llm_engine::core::TensorViewRef& view) {
    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::core::DeviceKind;
    using cpp_llm_engine::core::Status;

    const auto storage_status = view.validate(sizeof(float));
    if (!storage_status.is_ok()) return storage_status;
    if (view.contract.shape.dimensions.size() != 2) {
        return Status::error("tensor matrix view rank must be two");
    }
    if (view.contract.dtype != DType::Float32) {
        return Status::error("tensor matrix view dtype must be Float32");
    }
    if (view.contract.device != DeviceKind::CPU) {
        return Status::error("tensor matrix view device must be CPU");
    }
    if (!view.contract.is_contiguous_row_major()) {
        return Status::error("tensor matrix view must be contiguous row-major");
    }
    return Status::ok();
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.ready() || !rhs.ready() || lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return {};
    }

    Tensor out(lhs.rows, lhs.cols);
    const auto status =
        try_add_views(lhs.view_ref(), rhs.view_ref(), out.mutable_view_ref());
    if (!status.is_ok()) return {};
    return out;
}

cpp_llm_engine::core::Status try_add_views(
    const cpp_llm_engine::core::TensorViewRef& lhs,
    const cpp_llm_engine::core::TensorViewRef& rhs,
    const cpp_llm_engine::core::MutableTensorViewRef& output) {
    using cpp_llm_engine::core::Status;

    const auto lhs_status = validate_cpu_float32_matrix_view(lhs);
    if (!lhs_status.is_ok()) return lhs_status;
    const auto rhs_status = validate_cpu_float32_matrix_view(rhs);
    if (!rhs_status.is_ok()) return rhs_status;
    const auto output_status = validate_cpu_float32_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;
    if (lhs.contract.shape.dimensions != rhs.contract.shape.dimensions ||
        lhs.contract.shape.dimensions != output.contract.shape.dimensions) {
        return Status::error("residual add shape mismatch");
    }

    std::size_t elements = 0;
    const auto shape_status = lhs.contract.shape.checked_numel(&elements);
    if (!shape_status.is_ok()) return shape_status;
    const auto* lhs_data = static_cast<const float*>(lhs.storage_data) +
                           lhs.contract.storage_offset_elements;
    const auto* rhs_data = static_cast<const float*>(rhs.storage_data) +
                           rhs.contract.storage_offset_elements;
    std::vector<float> result(elements, 0.0f);
    for (std::size_t i = 0; i < elements; ++i) {
        result[i] = lhs_data[i] + rhs_data[i];
    }
    auto* output_data = static_cast<float*>(output.storage_data) +
                        output.contract.storage_offset_elements;
    std::copy(result.begin(), result.end(), output_data);
    return Status::ok();
}

Tensor transpose(const Tensor& tensor) {
    if (!tensor.ready()) {
        return {};
    }

    Tensor out(tensor.cols, tensor.rows);
    for (std::size_t row = 0; row < tensor.rows; ++row) {
        for (std::size_t col = 0; col < tensor.cols; ++col) {
            out.at(col, row) = tensor.at(row, col);
        }
    }
    return out;
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.ready() || !rhs.ready() || lhs.cols != rhs.rows) {
        return {};
    }

    Tensor out(lhs.rows, rhs.cols);
    for (std::size_t row = 0; row < lhs.rows; ++row) {
        for (std::size_t col = 0; col < rhs.cols; ++col) {
            float value = 0.0f;
            for (std::size_t k = 0; k < lhs.cols; ++k) {
                value += lhs.at(row, k) * rhs.at(k, col);
            }
            out.at(row, col) = value;
        }
    }
    return out;
}

Tensor slice_columns(const Tensor& tensor, std::size_t start_col, std::size_t width) {
    if (!tensor.ready() || start_col > tensor.cols || width > tensor.cols - start_col) {
        return {};
    }

    Tensor out(tensor.rows, width);
    for (std::size_t row = 0; row < tensor.rows; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            out.at(row, col) = tensor.at(row, start_col + col);
        }
    }
    return out;
}

Tensor concatenate_columns(const std::vector<Tensor>& parts) {
    if (parts.empty()) {
        return {};
    }

    const std::size_t rows = parts.front().rows;
    std::size_t cols = 0;
    for (const auto& part : parts) {
        if (!part.ready() || part.rows != rows ||
            part.cols > std::numeric_limits<std::size_t>::max() - cols) {
            return {};
        }
        cols += part.cols;
    }

    Tensor out(rows, cols);
    std::size_t offset = 0;
    for (const auto& part : parts) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < part.cols; ++col) {
                out.at(row, offset + col) = part.at(row, col);
            }
        }
        offset += part.cols;
    }
    return out;
}

void scale_in_place(Tensor& tensor, float factor) {
    if (!tensor.ready()) {
        return;
    }
    for (float& value : tensor.data) {
        value *= factor;
    }
}

}  // namespace chapter04_attention
