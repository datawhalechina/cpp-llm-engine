// Chapter 05：实现 Q3_K Linear 正确性参考与不展开完整权重的 CPU 路径。
#include "chapter05_quantization/q3k_linear.h"

#include "chapter05_quantization/q3k_dequantize.h"

#include <algorithm>
#include <array>
#include <limits>

namespace chapter05_quantization {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

cpp_llm_engine::core::Status validate(const cpp_llm_engine::core::LinearRequest& request) {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q3K) {
        return cpp_llm_engine::core::Status::error("Q3_K backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q3_K Linear buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 || request.cols % kQ3KBlockElements != 0) {
        return cpp_llm_engine::core::Status::error(
            "Q3_K Linear dimensions must be non-zero and cols must be a multiple of 256");
    }
    if (request.input_size != request.cols || request.output_size < request.rows) {
        return cpp_llm_engine::core::Status::error("Q3_K Linear buffer sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ3KBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        request.rows * blocks_per_row > std::numeric_limits<std::size_t>::max() / kQ3KBlockBytes) {
        return cpp_llm_engine::core::Status::error("Q3_K Linear payload is too large");
    }
    if (request.weight_bytes != request.rows * blocks_per_row * kQ3KBlockBytes) {
        return cpp_llm_engine::core::Status::error("Q3_K Linear byte count does not match shape");
    }
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace

bool Q3KLinearReference::forward(const std::uint8_t* weights, std::size_t weight_bytes,
                                 std::size_t rows, std::size_t cols,
                                 const std::vector<float>& input,
                                 std::vector<float>* output, std::string* error) {
    if (weights == nullptr || output == nullptr) return fail(error, "Q3_K Linear input and output must not be null");
    std::vector<float> decoded;
    if (!dequantize_q3_k(weights, weight_bytes, rows * cols, &decoded, error)) return false;
    if (rows == 0 || cols == 0 || input.size() != cols) return fail(error, "Q3_K Linear shape does not match input");
    output->assign(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            (*output)[row] += decoded[row * cols + col] * input[col];
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

cpp_llm_engine::core::BackendKind Q3KLinearFusedBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q3KLinearFusedBackend::name() const { return "q3k-fused-cpu"; }

cpp_llm_engine::core::Status Q3KLinearFusedBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    const auto status = validate(request);
    if (!status.is_ok()) return status;
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    const std::size_t blocks_per_row = request.cols / kQ3KBlockElements;
    std::array<float, kQ3KBlockElements> block_values{};
    for (std::size_t row = 0; row < request.rows; ++row) {
        float sum = 0.0f;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            decode_q3_k_block(weights + (row * blocks_per_row + block) * kQ3KBlockBytes,
                              block_values.data());
            for (std::size_t value = 0; value < kQ3KBlockElements; ++value) {
                sum += block_values[value] * request.input[block * kQ3KBlockElements + value];
            }
        }
        request.output[row] = sum;
    }
    return cpp_llm_engine::core::Status::ok();
}

cpp_llm_engine::core::Status Q3KLinearFusedBackend::compute_batch(
    const cpp_llm_engine::core::LinearBatchRequest& request) const {
    using cpp_llm_engine::core::Status;
    if (request.batch_size == 0 || request.batch_size >
            std::numeric_limits<std::size_t>::max() / request.cols ||
        request.batch_size >
            std::numeric_limits<std::size_t>::max() / request.rows ||
        request.input_size != request.batch_size * request.cols ||
        request.output_size < request.batch_size * request.rows) {
        return Status::error("Q3_K Linear batch buffer sizes do not match shape");
    }
    const cpp_llm_engine::core::LinearRequest first{
        request.weights, request.weight_bytes, request.input, request.cols,
        request.output, request.rows, request.rows, request.cols,
        request.weight_format};
    const auto status = validate(first);
    if (!status.is_ok()) return status;

    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    const std::size_t blocks_per_row = request.cols / kQ3KBlockElements;
    std::array<float, kQ3KBlockElements> block_values{};
    std::fill(request.output,
              request.output + request.batch_size * request.rows, 0.0F);
    for (std::size_t row = 0; row < request.rows; ++row) {
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            decode_q3_k_block(
                weights + (row * blocks_per_row + block) * kQ3KBlockBytes,
                block_values.data());
            for (std::size_t batch = 0; batch < request.batch_size; ++batch) {
                float sum = 0.0F;
                const float* input = request.input + batch * request.cols +
                    block * kQ3KBlockElements;
                for (std::size_t value = 0; value < kQ3KBlockElements; ++value) {
                    sum += block_values[value] * input[value];
                }
                request.output[batch * request.rows + row] += sum;
            }
        }
    }
    return Status::ok();
}

const Q3KLinearFusedBackend& q3k_fused_backend() {
    static const Q3KLinearFusedBackend backend;
    return backend;
}

}  // namespace chapter05_quantization
