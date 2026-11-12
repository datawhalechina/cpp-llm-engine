// Core：量化 block 布局契约
// 职责：统一 CUDA/CPU 共用量化格式的 block 尺寸，并安全计算 row-major 权重字节数。
// 契约：每一行必须由完整 block 组成；本文件不拥有 payload，也不执行解码。
// 验证：core contract、Chapter05 CPU 测试和 CUDA Quantized Linear 测试。
#pragma once

#include "linear_contract.h"
#include "status.h"

#include <cstddef>
#include <limits>

namespace cpp_llm_engine::core {

constexpr std::size_t kQuantizedBlockElements = 256;
constexpr std::size_t kQ4KBlockBytes = 144;
constexpr std::size_t kQ6KBlockBytes = 210;
constexpr std::size_t kQ8_0BlockElements = 32;
constexpr std::size_t kQ8_0BlockBytes = 34;
constexpr std::size_t kQ2KBlockBytes = 84;
constexpr std::size_t kQ3KBlockBytes = 110;
constexpr std::size_t kIQ4NLBlockElements = 32;
constexpr std::size_t kIQ4NLBlockBytes = 18;

inline Status quantized_block_layout(WeightFormat format,
                                     std::size_t* elements,
                                     std::size_t* bytes) {
    if (elements == nullptr || bytes == nullptr) {
        return Status::error("quantized block layout output is null");
    }
    if (format == WeightFormat::Q4K) {
        *elements = kQuantizedBlockElements;
        *bytes = kQ4KBlockBytes;
        return Status::ok();
    }
    if (format == WeightFormat::Q6K) {
        *elements = kQuantizedBlockElements;
        *bytes = kQ6KBlockBytes;
        return Status::ok();
    }
    if (format == WeightFormat::Q8_0) {
        *elements = kQ8_0BlockElements;
        *bytes = kQ8_0BlockBytes;
        return Status::ok();
    }
    if (format == WeightFormat::Q2K) {
        *elements = kQuantizedBlockElements;
        *bytes = kQ2KBlockBytes;
        return Status::ok();
    }
    if (format == WeightFormat::Q3K) {
        *elements = kQuantizedBlockElements;
        *bytes = kQ3KBlockBytes;
        return Status::ok();
    }
    if (format == WeightFormat::IQ4_NL) {
        *elements = kIQ4NLBlockElements;
        *bytes = kIQ4NLBlockBytes;
        return Status::ok();
    }
    return Status::error("unsupported quantized block format");
}

inline Status quantized_block_bytes(WeightFormat format, std::size_t* result) {
    std::size_t elements = 0;
    return quantized_block_layout(format, &elements, result);
}

inline Status checked_quantized_matrix_bytes(
    WeightFormat format,
    std::size_t rows,
    std::size_t cols,
    std::size_t* result) {
    if (result == nullptr) {
        return Status::error("quantized matrix-byte output is null");
    }
    if (rows == 0 || cols == 0) {
        return Status::error("quantized matrix dimensions must be non-zero");
    }
    std::size_t block_elements = 0;
    std::size_t block_bytes = 0;
    const Status layout_status = quantized_block_layout(
        format, &block_elements, &block_bytes);
    if (!layout_status.is_ok()) return layout_status;
    if (cols % block_elements != 0) {
        return Status::error(
            "quantized matrix cols must be a multiple of the format block size");
    }
    const std::size_t blocks_per_row = cols / block_elements;
    if (rows > std::numeric_limits<std::size_t>::max() / blocks_per_row) {
        return Status::error("quantized matrix block count overflows host size");
    }
    const std::size_t blocks = rows * blocks_per_row;
    if (blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
        return Status::error("quantized matrix byte count overflows host size");
    }
    *result = blocks * block_bytes;
    return Status::ok();
}

}  // namespace cpp_llm_engine::core
