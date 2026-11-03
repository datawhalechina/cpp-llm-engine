// Core：Tensor 元数据契约
// 职责：定义 shape、stride、storage、dtype 和 device 的公共表示及安全检查入口。
// 契约：只描述对象和布局，不拥有具体章节的 storage 或 kernel。
// 验证：code/core/tests/core_contract_test.cpp。
#pragma once

#include "status.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cpp_llm_engine::core {

enum class DType : std::uint8_t {
    Unknown = 0,
    Float32,
    Float16,
    Int8,
    Q4K,
    Q6K,
};

enum class DeviceKind : std::uint8_t {
    CPU = 0,
    CUDA,
};

struct Shape final {
    std::vector<std::size_t> dimensions;

    Status checked_numel(std::size_t* result) const {
        if (result == nullptr || dimensions.empty()) {
            return Status::error("shape output is null or dimensions are empty");
        }
        std::size_t value = 1;
        for (const auto dimension : dimensions) {
            if (dimension != 0 && value > std::numeric_limits<std::size_t>::max() / dimension) {
                return Status::error("shape element count overflows host size");
            }
            value *= dimension;
        }
        *result = value;
        return Status::ok();
    }
};

struct TensorContract final {
    Shape shape;
    std::vector<std::size_t> strides;
    DType dtype = DType::Unknown;
    DeviceKind device = DeviceKind::CPU;
    std::size_t storage_offset_elements = 0;
    std::size_t storage_bytes = 0;

    bool is_contiguous_row_major() const {
        if (shape.dimensions.empty() ||
            strides.size() != shape.dimensions.size()) {
            return false;
        }
        std::size_t expected_stride = 1;
        for (std::size_t axis = shape.dimensions.size(); axis-- > 0;) {
            const auto dimension = shape.dimensions[axis];
            if (dimension == 0) return false;
            if (dimension != 1 && strides[axis] != expected_stride) {
                return false;
            }
            if (expected_stride >
                std::numeric_limits<std::size_t>::max() / dimension) {
                return false;
            }
            expected_stride *= dimension;
        }
        return true;
    }

    Status checked_required_elements(std::size_t* result) const {
        if (result == nullptr) {
            return Status::error("tensor required-elements output is null");
        }
        if (shape.dimensions.empty() || strides.size() != shape.dimensions.size()) {
            return Status::error("tensor shape and strides must have the same non-zero rank");
        }
        std::size_t last = storage_offset_elements;
        for (std::size_t axis = 0; axis < shape.dimensions.size(); ++axis) {
            const auto dimension = shape.dimensions[axis];
            if (dimension == 0) {
                return Status::error("tensor dimensions must be non-zero");
            }
            const auto extent = dimension - 1;
            if (extent != 0 && strides[axis] >
                    std::numeric_limits<std::size_t>::max() / extent) {
                return Status::error("tensor stride extent overflows host size");
            }
            const auto axis_offset = extent * strides[axis];
            if (last > std::numeric_limits<std::size_t>::max() - axis_offset) {
                return Status::error("tensor storage offset overflows host size");
            }
            last += axis_offset;
        }
        if (last == std::numeric_limits<std::size_t>::max()) {
            return Status::error("tensor required element count overflows host size");
        }
        *result = last + 1;
        return Status::ok();
    }

    Status validate_element_view(std::size_t element_size_bytes) const {
        if (dtype == DType::Unknown) {
            return Status::error("tensor dtype must be known");
        }
        if (element_size_bytes == 0) {
            return Status::error("tensor element size must be non-zero");
        }
        std::size_t required_elements = 0;
        const auto status = checked_required_elements(&required_elements);
        if (!status.is_ok()) return status;
        if (required_elements >
                std::numeric_limits<std::size_t>::max() / element_size_bytes) {
            return Status::error("tensor required byte count overflows host size");
        }
        if (storage_bytes < required_elements * element_size_bytes) {
            return Status::error("tensor view exceeds storage bytes");
        }
        return Status::ok();
    }
};

}  // namespace cpp_llm_engine::core
