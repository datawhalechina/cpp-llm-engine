// Chapter 05：Q8_0 Linear CPU 实现
// 职责：reference 先完整解码；fused backend 在每个 32-value block 上直接累加。
// 契约：scale 为 little-endian FP16，quant 为 32 个有符号 int8。
// 验证：chapter05_quantization_test 与 SmolVLM Q8_0 Decoder 路径。
#include "chapter05_quantization/q8_0_linear.h"

#include "chapter05_quantization/q8_0_dequantize.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace chapter05_quantization {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

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

bool validate_shape(std::size_t weight_bytes,
                    std::size_t rows,
                    std::size_t cols,
                    const std::vector<float>& input,
                    std::string* error) {
    if (rows == 0 || cols == 0 || cols % kQ8_0BlockElements != 0) {
        return fail(error,
                    "Q8_0 Linear dimensions must be non-zero and cols must be a multiple of 32");
    }
    if (input.size() != cols) return fail(error, "Q8_0 Linear input width does not match cols");
    const std::size_t blocks_per_row = cols / kQ8_0BlockElements;
    if (rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        rows * blocks_per_row > std::numeric_limits<std::size_t>::max() / kQ8_0BlockBytes) {
        return fail(error, "Q8_0 Linear payload is too large");
    }
    if (weight_bytes != rows * blocks_per_row * kQ8_0BlockBytes) {
        return fail(error, "Q8_0 Linear byte count does not match shape");
    }
    return true;
}

}  // namespace

bool Q8_0LinearReference::forward(const std::uint8_t* weights,
                                  std::size_t weight_bytes,
                                  std::size_t rows,
                                  std::size_t cols,
                                  const std::vector<float>& input,
                                  std::vector<float>* output,
                                  std::string* error) {
    if (weights == nullptr || output == nullptr) {
        return fail(error, "Q8_0 Linear input and output must not be null");
    }
    if (!validate_shape(weight_bytes, rows, cols, input, error)) return false;
    std::vector<float> decoded;
    if (!dequantize_q8_0(weights, weight_bytes, rows * cols, &decoded, error)) return false;
    output->assign(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += decoded[row * cols + col] * input[col];
        }
        (*output)[row] = sum;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool Q8_0LinearFused::forward(const std::uint8_t* weights,
                              std::size_t weight_bytes,
                              std::size_t rows,
                              std::size_t cols,
                              const std::vector<float>& input,
                              std::vector<float>* output,
                              std::string* error) {
    if (weights == nullptr || output == nullptr) {
        return fail(error, "Q8_0 Linear input and output must not be null");
    }
    if (!validate_shape(weight_bytes, rows, cols, input, error)) return false;
    output->assign(rows, 0.0f);
    const cpp_llm_engine::core::LinearRequest request{
        weights, weight_bytes, input.data(), input.size(), output->data(), output->size(),
        rows, cols, cpp_llm_engine::core::WeightFormat::Q8_0};
    const auto status = q8_0_fused_backend().compute(request);
    if (!status.is_ok()) return fail(error, status.message());
    if (error != nullptr) error->clear();
    return true;
}

cpp_llm_engine::core::BackendKind Q8_0LinearFusedBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q8_0LinearFusedBackend::name() const {
    return "q8_0-fused-cpu";
}

cpp_llm_engine::core::Status Q8_0LinearFusedBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q8_0) {
        return cpp_llm_engine::core::Status::error(
            "Q8_0 backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q8_0 Linear buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 ||
        request.cols % kQ8_0BlockElements != 0) {
        return cpp_llm_engine::core::Status::error(
            "Q8_0 Linear dimensions must be non-zero and cols must be a multiple of 32");
    }
    if (request.input_size != request.cols || request.output_size < request.rows) {
        return cpp_llm_engine::core::Status::error(
            "Q8_0 Linear buffer sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ8_0BlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        request.rows * blocks_per_row >
            std::numeric_limits<std::size_t>::max() / kQ8_0BlockBytes) {
        return cpp_llm_engine::core::Status::error("Q8_0 Linear payload is too large");
    }
    if (request.weight_bytes != request.rows * blocks_per_row * kQ8_0BlockBytes) {
        return cpp_llm_engine::core::Status::error(
            "Q8_0 Linear byte count does not match shape");
    }

    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    for (std::size_t row = 0; row < request.rows; ++row) {
        float sum = 0.0f;
        const auto* row_weights = weights + row * blocks_per_row * kQ8_0BlockBytes;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto* packed = row_weights + block * kQ8_0BlockBytes;
            std::uint16_t scale_bits = 0;
            std::memcpy(&scale_bits, packed, sizeof(scale_bits));
            const float scale = half_to_float(scale_bits);
            const std::size_t input_offset = block * kQ8_0BlockElements;
            for (std::size_t value = 0; value < kQ8_0BlockElements; ++value) {
                std::int8_t quant = 0;
                std::memcpy(&quant, packed + sizeof(scale_bits) + value, sizeof(quant));
                sum += scale * static_cast<float>(quant) * request.input[input_offset + value];
            }
        }
        request.output[row] = sum;
    }
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::Status Q8_0LinearFusedBackend::compute_batch(
    const cpp_llm_engine::core::LinearBatchRequest& request) const {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q8_0) {
        return cpp_llm_engine::core::Status::error(
            "Q8_0 batch backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q8_0 Linear batch buffers must not be null");
    }
    if (request.batch_size == 0 || request.rows == 0 || request.cols == 0 ||
        request.cols % kQ8_0BlockElements != 0 ||
        request.batch_size > std::numeric_limits<std::size_t>::max() / request.cols ||
        request.batch_size > std::numeric_limits<std::size_t>::max() / request.rows ||
        request.input_size != request.batch_size * request.cols ||
        request.output_size < request.batch_size * request.rows) {
        return cpp_llm_engine::core::Status::error(
            "Q8_0 Linear batch sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ8_0BlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        request.rows * blocks_per_row >
            std::numeric_limits<std::size_t>::max() / kQ8_0BlockBytes ||
        request.weight_bytes != request.rows * blocks_per_row * kQ8_0BlockBytes) {
        return cpp_llm_engine::core::Status::error(
            "Q8_0 Linear batch weight bytes do not match shape");
    }

    std::fill(request.output, request.output + request.batch_size * request.rows, 0.0f);
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    for (std::size_t row = 0; row < request.rows; ++row) {
        const auto* row_weights = weights + row * blocks_per_row * kQ8_0BlockBytes;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto* packed = row_weights + block * kQ8_0BlockBytes;
            std::uint16_t scale_bits = 0;
            std::memcpy(&scale_bits, packed, sizeof(scale_bits));
            const float scale = half_to_float(scale_bits);
            const std::size_t input_offset = block * kQ8_0BlockElements;
            for (std::size_t value = 0; value < kQ8_0BlockElements; ++value) {
                std::int8_t quant = 0;
                std::memcpy(&quant, packed + sizeof(scale_bits) + value, sizeof(quant));
                const float weight = scale * static_cast<float>(quant);
                for (std::size_t batch = 0; batch < request.batch_size; ++batch) {
                    request.output[batch * request.rows + row] += weight *
                        request.input[batch * request.cols + input_offset + value];
                }
            }
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

const Q8_0LinearFusedBackend& q8_0_fused_backend() {
    static const Q8_0LinearFusedBackend backend;
    return backend;
}

}  // namespace chapter05_quantization
