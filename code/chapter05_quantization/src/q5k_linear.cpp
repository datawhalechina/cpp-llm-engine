// Chapter 05：实现 GGML Q5_K 单行与多行 CPU Linear。
// 批量路径每次解码一个权重块，并复用于同组的全部输入行。
#include "chapter05_quantization/q5k_linear.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace chapter05_quantization {
namespace {

float fp16(std::uint16_t half) {
    const std::uint32_t sign = (half & 0x8000u) << 16;
    const std::uint32_t exponent = (half >> 10) & 31u;
    const std::uint32_t mantissa = half & 1023u;
    std::uint32_t raw = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            raw = sign;
        } else {
            auto normalized = mantissa;
            std::uint32_t adjusted = 113;
            while ((normalized & 1024u) == 0) {
                normalized <<= 1;
                --adjusted;
            }
            raw = sign | (adjusted << 23) | ((normalized & 1023u) << 13);
        }
    } else if (exponent == 31) {
        raw = sign | 0x7f800000u | (mantissa << 13);
    } else {
        raw = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0F;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

void scale_min(int group, const std::uint8_t* packed,
               std::uint8_t* scale, std::uint8_t* minimum) {
    if (group < 4) {
        *scale = packed[group] & 63;
        *minimum = packed[group + 4] & 63;
    } else {
        *scale = (packed[group + 4] & 15) |
            ((packed[group - 4] >> 6) << 4);
        *minimum = (packed[group + 4] >> 4) |
            ((packed[group] >> 6) << 4);
    }
}

cpp_llm_engine::core::Status validate(
    const void* weights, std::size_t weight_bytes, std::size_t rows,
    std::size_t cols, cpp_llm_engine::core::WeightFormat format) {
    using cpp_llm_engine::core::Status;
    if (format != cpp_llm_engine::core::WeightFormat::Q5K) {
        return Status::error("Q5_K backend received an unsupported weight format");
    }
    if (weights == nullptr || rows == 0 || cols == 0 || cols % 256 != 0) {
        return Status::error("Q5_K Linear shape or weight payload is invalid");
    }
    const std::size_t blocks_per_row = cols / kQ5KBlockElements;
    if (rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        weight_bytes != rows * blocks_per_row * kQ5KBlockBytes) {
        return Status::error("Q5_K Linear byte count does not match shape");
    }
    return Status::ok();
}

void decode_block(const std::uint8_t* packed, float* output) {
    std::uint16_t delta_bits = 0;
    std::uint16_t minimum_bits = 0;
    std::memcpy(&delta_bits, packed, sizeof(delta_bits));
    std::memcpy(&minimum_bits, packed + 2, sizeof(minimum_bits));
    const float delta = fp16(delta_bits);
    const float minimum_delta = fp16(minimum_bits);
    const auto* scales = packed + 4;
    const auto* high = packed + 16;
    const auto* low = packed + 48;
    int scale_index = 0;
    std::uint8_t high_mask_low = 1;
    std::uint8_t high_mask_high = 2;
    std::size_t index = 0;
    for (int group = 0; group < 256; group += 64) {
        std::uint8_t scale_low = 0;
        std::uint8_t minimum_low = 0;
        std::uint8_t scale_high = 0;
        std::uint8_t minimum_high = 0;
        scale_min(scale_index, scales, &scale_low, &minimum_low);
        scale_min(scale_index + 1, scales, &scale_high, &minimum_high);
        for (int lane = 0; lane < 32; ++lane) {
            const int quant = (low[lane] & 15) +
                ((high[lane] & high_mask_low) != 0 ? 16 : 0);
            output[index++] = delta * scale_low * quant -
                minimum_delta * minimum_low;
        }
        for (int lane = 0; lane < 32; ++lane) {
            const int quant = (low[lane] >> 4) +
                ((high[lane] & high_mask_high) != 0 ? 16 : 0);
            output[index++] = delta * scale_high * quant -
                minimum_delta * minimum_high;
        }
        low += 32;
        scale_index += 2;
        high_mask_low <<= 2;
        high_mask_high <<= 2;
    }
}

}  // namespace

cpp_llm_engine::core::BackendKind Q5KLinearBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q5KLinearBackend::name() const {
    return "q5k-scalar-cpu";
}

cpp_llm_engine::core::Status Q5KLinearBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    using cpp_llm_engine::core::Status;
    const auto status = validate(
        request.weights, request.weight_bytes, request.rows, request.cols,
        request.weight_format);
    if (!status.is_ok()) return status;
    if (request.input == nullptr || request.output == nullptr ||
        request.input_size != request.cols || request.output_size < request.rows) {
        return Status::error("Q5_K Linear buffers do not match shape");
    }
    const cpp_llm_engine::core::LinearBatchRequest batch{
        request.weights, request.weight_bytes, request.input, request.input_size,
        request.output, request.output_size, 1, request.rows, request.cols,
        request.weight_format};
    return compute_batch(batch);
}

cpp_llm_engine::core::Status Q5KLinearBackend::compute_batch(
    const cpp_llm_engine::core::LinearBatchRequest& request) const {
    using cpp_llm_engine::core::Status;
    const auto status = validate(
        request.weights, request.weight_bytes, request.rows, request.cols,
        request.weight_format);
    if (!status.is_ok()) return status;
    if (request.input == nullptr || request.output == nullptr ||
        request.batch_size == 0 ||
        request.batch_size > std::numeric_limits<std::size_t>::max() / request.cols ||
        request.batch_size > std::numeric_limits<std::size_t>::max() / request.rows ||
        request.input_size != request.batch_size * request.cols ||
        request.output_size < request.batch_size * request.rows) {
        return Status::error("Q5_K Linear batch buffers do not match shape");
    }
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    const std::size_t blocks_per_row = request.cols / kQ5KBlockElements;
    std::array<float, kQ5KBlockElements> decoded{};
    std::fill(request.output,
              request.output + request.batch_size * request.rows, 0.0F);
    for (std::size_t row = 0; row < request.rows; ++row) {
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            decode_block(
                weights + (row * blocks_per_row + block) * kQ5KBlockBytes,
                decoded.data());
            for (std::size_t batch = 0; batch < request.batch_size; ++batch) {
                float sum = 0.0F;
                const float* input = request.input + batch * request.cols +
                    block * kQ5KBlockElements;
                for (std::size_t value = 0; value < kQ5KBlockElements; ++value) {
                    sum += decoded[value] * input[value];
                }
                request.output[batch * request.rows + row] += sum;
            }
        }
    }
    return Status::ok();
}

const Q5KLinearBackend& q5k_backend() {
    static const Q5KLinearBackend backend;
    return backend;
}

}  // namespace chapter05_quantization
