// Chapter 05：实现 Q4_K scalar fused decode-and-dot kernel。
// 职责：校验 Linear contract 后直接从 packed block 累加输出。
// 验证入口：chapter05_quantization_test。
#include "chapter05_quantization/q4k_kernel.h"

#include "chapter05_quantization/q4k_dequantize.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace chapter05_quantization {
namespace {

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::uint32_t normalized = mantissa;
            std::uint32_t exp = 127 - 14;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            result = sign | (exp << 23) | ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

void get_scale_min(const std::uint8_t* scales,
                   int index,
                   std::uint8_t* scale,
                   std::uint8_t* minimum) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

}  // namespace

cpp_llm_engine::core::Status q4k_fused_cpu_kernel(
    const cpp_llm_engine::core::LinearRequest& request) {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q4K) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q4_K Linear buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 ||
        request.cols % kQ4KBlockElements != 0) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K Linear dimensions must be non-zero and cols must be a multiple of 256");
    }
    if (request.input_size != request.cols || request.output_size < request.rows) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K Linear buffer sizes do not match shape");
    }

    const std::size_t blocks_per_row = request.cols / kQ4KBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row) {
        return cpp_llm_engine::core::Status::error("Q4_K Linear payload is too large");
    }
    const std::size_t block_count = request.rows * blocks_per_row;
    if (block_count > std::numeric_limits<std::size_t>::max() / kQ4KBlockBytes) {
        return cpp_llm_engine::core::Status::error("Q4_K Linear payload is too large");
    }
    if (request.weight_bytes != block_count * kQ4KBlockBytes) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K Linear byte count does not match shape");
    }

    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    for (std::size_t row = 0; row < request.rows; ++row) {
        float sum = 0.0f;
        const std::uint8_t* row_weights =
            weights + row * blocks_per_row * kQ4KBlockBytes;
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const std::uint8_t* block =
                row_weights + block_index * kQ4KBlockBytes;
            std::uint16_t d_bits = 0;
            std::uint16_t dmin_bits = 0;
            std::memcpy(&d_bits, block, sizeof(d_bits));
            std::memcpy(&dmin_bits, block + 2, sizeof(dmin_bits));
            const float d = half_to_float(d_bits);
            const float minimum = half_to_float(dmin_bits);
            const auto* scales = block + 4;
            const auto* quants = block + 16;
            const std::size_t input_offset = block_index * kQ4KBlockElements;
            int scale_index = 0;
            for (std::size_t group = 0; group < kQ4KBlockElements; group += 64) {
                std::uint8_t scale = 0;
                std::uint8_t min_value = 0;
                get_scale_min(scales, scale_index, &scale, &min_value);
                const float d1 = d * scale;
                const float m1 = minimum * min_value;
                get_scale_min(scales, scale_index + 1, &scale, &min_value);
                const float d2 = d * scale;
                const float m2 = minimum * min_value;
                for (int i = 0; i < 32; ++i) {
                    sum += (d1 * (quants[i] & 0x0f) - m1) *
                           request.input[input_offset + group + i];
                    sum += (d2 * (quants[i] >> 4) - m2) *
                           request.input[input_offset + group + 32 + i];
                }
                quants += 32;
                scale_index += 2;
            }
        }
        request.output[row] = sum;
    }
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::Status q4k_fused_cpu_batch_kernel(
    const cpp_llm_engine::core::LinearBatchRequest& request) {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q4K) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K batch backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q4_K Linear batch buffers must not be null");
    }
    if (request.batch_size == 0 || request.rows == 0 || request.cols == 0 ||
        request.cols % kQ4KBlockElements != 0 ||
        request.batch_size > std::numeric_limits<std::size_t>::max() / request.cols ||
        request.batch_size > std::numeric_limits<std::size_t>::max() / request.rows ||
        request.input_size != request.batch_size * request.cols ||
        request.output_size < request.batch_size * request.rows) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K Linear batch sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ4KBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        request.rows * blocks_per_row >
            std::numeric_limits<std::size_t>::max() / kQ4KBlockBytes ||
        request.weight_bytes != request.rows * blocks_per_row * kQ4KBlockBytes) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K Linear batch weight bytes do not match shape");
    }

    std::fill(request.output, request.output + request.batch_size * request.rows, 0.0f);
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    for (std::size_t row = 0; row < request.rows; ++row) {
        const auto* row_weights = weights + row * blocks_per_row * kQ4KBlockBytes;
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const auto* block = row_weights + block_index * kQ4KBlockBytes;
            std::uint16_t d_bits = 0;
            std::uint16_t dmin_bits = 0;
            std::memcpy(&d_bits, block, sizeof(d_bits));
            std::memcpy(&dmin_bits, block + 2, sizeof(dmin_bits));
            const float d = half_to_float(d_bits);
            const float minimum = half_to_float(dmin_bits);
            const auto* scales = block + 4;
            const auto* quants = block + 16;
            const std::size_t input_offset = block_index * kQ4KBlockElements;
            int scale_index = 0;
            for (std::size_t group = 0; group < kQ4KBlockElements; group += 64) {
                std::uint8_t scale = 0;
                std::uint8_t min_value = 0;
                get_scale_min(scales, scale_index, &scale, &min_value);
                const float d1 = d * scale;
                const float m1 = minimum * min_value;
                get_scale_min(scales, scale_index + 1, &scale, &min_value);
                const float d2 = d * scale;
                const float m2 = minimum * min_value;
                for (int i = 0; i < 32; ++i) {
                    const float weight1 = d1 * (quants[i] & 0x0f) - m1;
                    const float weight2 = d2 * (quants[i] >> 4) - m2;
                    for (std::size_t batch = 0; batch < request.batch_size; ++batch) {
                        const auto* input = request.input + batch * request.cols;
                        auto& sum = request.output[batch * request.rows + row];
                        sum += weight1 * input[input_offset + group + i];
                        sum += weight2 * input[input_offset + group + 32 + i];
                    }
                }
                quants += 32;
                scale_index += 2;
            }
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace chapter05_quantization
