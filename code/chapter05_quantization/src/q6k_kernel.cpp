// 中文：实现 Q6_K fused CPU Linear，保持与 reference dequant 相同的位解码顺序。
#include "chapter05_quantization/q6k_kernel.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "chapter05_quantization/q6k_dequantize.h"

namespace chapter05_quantization {
namespace {

struct Q6KBlock {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    std::uint16_t d;
};

static_assert(sizeof(Q6KBlock) == kQ6KBlockBytes, "unexpected Q6_K block size");

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

}  // namespace

cpp_llm_engine::core::Status q6k_fused_cpu_kernel(
    const cpp_llm_engine::core::LinearRequest& request) {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q6K) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q6_K Linear buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 || request.cols % kQ6KBlockElements != 0) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K Linear dimensions must be non-zero and cols must be a multiple of 256");
    }
    if (request.input_size != request.cols || request.output_size < request.rows) {
        return cpp_llm_engine::core::Status::error("Q6_K Linear buffer sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ6KBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row) {
        return cpp_llm_engine::core::Status::error("Q6_K Linear payload is too large");
    }
    const std::size_t block_count = request.rows * blocks_per_row;
    if (block_count > std::numeric_limits<std::size_t>::max() / kQ6KBlockBytes) {
        return cpp_llm_engine::core::Status::error("Q6_K Linear payload is too large");
    }
    if (request.weight_bytes != block_count * kQ6KBlockBytes) {
        return cpp_llm_engine::core::Status::error("Q6_K Linear byte count does not match shape");
    }

    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    for (std::size_t row = 0; row < request.rows; ++row) {
        float sum = 0.0f;
        const auto* row_weights = weights + row * blocks_per_row * kQ6KBlockBytes;
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            Q6KBlock block{};
            std::memcpy(&block, row_weights + block_index * kQ6KBlockBytes, sizeof(block));
            const float d = half_to_float(block.d);
            const std::size_t input_offset = block_index * kQ6KBlockElements;
            for (std::size_t half = 0; half < kQ6KBlockElements; half += 128) {
                for (int l = 0; l < 32; ++l) {
                    const std::size_t ql_offset = half / 2;
                    const std::size_t qh_offset = half / 4;
                    const std::size_t scale_offset = half / 16;
                    const int scale_index = l / 16;
                    const int q1 = static_cast<int>((block.ql[ql_offset + l] & 0x0f) | (((block.qh[qh_offset + l] >> 0) & 0x03) << 4)) - 32;
                    const int q2 = static_cast<int>((block.ql[ql_offset + l + 32] & 0x0f) | (((block.qh[qh_offset + l] >> 2) & 0x03) << 4)) - 32;
                    const int q3 = static_cast<int>(((block.ql[ql_offset + l] >> 4) & 0x0f) | (((block.qh[qh_offset + l] >> 4) & 0x03) << 4)) - 32;
                    const int q4 = static_cast<int>(((block.ql[ql_offset + l + 32] >> 4) & 0x0f) | (((block.qh[qh_offset + l] >> 6) & 0x03) << 4)) - 32;
                    sum += d * block.scales[scale_offset + scale_index + 0] * q1 * request.input[input_offset + half + l];
                    sum += d * block.scales[scale_offset + scale_index + 2] * q2 * request.input[input_offset + half + l + 32];
                    sum += d * block.scales[scale_offset + scale_index + 4] * q3 * request.input[input_offset + half + l + 64];
                    sum += d * block.scales[scale_offset + scale_index + 6] * q4 * request.input[input_offset + half + l + 96];
                }
            }
        }
        request.output[row] = sum;
    }
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::Status q6k_fused_cpu_batch_kernel(
    const cpp_llm_engine::core::LinearBatchRequest& request) {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q6K) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K batch backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q6_K Linear batch buffers must not be null");
    }
    if (request.batch_size == 0 || request.rows == 0 || request.cols == 0 ||
        request.cols % kQ6KBlockElements != 0 ||
        request.batch_size > std::numeric_limits<std::size_t>::max() / request.cols ||
        request.batch_size > std::numeric_limits<std::size_t>::max() / request.rows ||
        request.input_size != request.batch_size * request.cols ||
        request.output_size < request.batch_size * request.rows) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K Linear batch sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ6KBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        request.rows * blocks_per_row >
            std::numeric_limits<std::size_t>::max() / kQ6KBlockBytes ||
        request.weight_bytes != request.rows * blocks_per_row * kQ6KBlockBytes) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K Linear batch weight bytes do not match shape");
    }

    std::fill(request.output, request.output + request.batch_size * request.rows, 0.0f);
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    for (std::size_t row = 0; row < request.rows; ++row) {
        const auto* row_weights = weights + row * blocks_per_row * kQ6KBlockBytes;
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            Q6KBlock block{};
            std::memcpy(&block, row_weights + block_index * kQ6KBlockBytes, sizeof(block));
            const float d = half_to_float(block.d);
            const std::size_t input_offset = block_index * kQ6KBlockElements;
            for (std::size_t half = 0; half < kQ6KBlockElements; half += 128) {
                for (int l = 0; l < 32; ++l) {
                    const std::size_t ql_offset = half / 2;
                    const std::size_t qh_offset = half / 4;
                    const std::size_t scale_offset = half / 16;
                    const int scale_index = l / 16;
                    const int q1 = static_cast<int>((block.ql[ql_offset + l] & 0x0f) | (((block.qh[qh_offset + l] >> 0) & 0x03) << 4)) - 32;
                    const int q2 = static_cast<int>((block.ql[ql_offset + l + 32] & 0x0f) | (((block.qh[qh_offset + l] >> 2) & 0x03) << 4)) - 32;
                    const int q3 = static_cast<int>(((block.ql[ql_offset + l] >> 4) & 0x0f) | (((block.qh[qh_offset + l] >> 4) & 0x03) << 4)) - 32;
                    const int q4 = static_cast<int>(((block.ql[ql_offset + l + 32] >> 4) & 0x0f) | (((block.qh[qh_offset + l] >> 6) & 0x03) << 4)) - 32;
                    const float w1 = d * block.scales[scale_offset + scale_index + 0] * q1;
                    const float w2 = d * block.scales[scale_offset + scale_index + 2] * q2;
                    const float w3 = d * block.scales[scale_offset + scale_index + 4] * q3;
                    const float w4 = d * block.scales[scale_offset + scale_index + 6] * q4;
                    for (std::size_t batch = 0; batch < request.batch_size; ++batch) {
                        const auto* input = request.input + batch * request.cols;
                        auto& sum = request.output[batch * request.rows + row];
                        sum += w1 * input[input_offset + half + l];
                        sum += w2 * input[input_offset + half + l + 32];
                        sum += w3 * input[input_offset + half + l + 64];
                        sum += w4 * input[input_offset + half + l + 96];
                    }
                }
            }
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace chapter05_quantization
