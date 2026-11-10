// Chapter 05：实现 Q5_1 Linear 正确性参考与不展开完整权重的 CPU 路径。
#include "chapter05_quantization/q5_1_linear.h"

#include "chapter05_quantization/q5_1_dequantize.h"

#include <array>
#include <limits>

namespace chapter05_quantization {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

cpp_llm_engine::core::Status validate(const cpp_llm_engine::core::LinearRequest& request) {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q5_1) {
        return cpp_llm_engine::core::Status::error("Q5_1 backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q5_1 Linear buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 || request.cols % kQ5_1BlockElements != 0) {
        return cpp_llm_engine::core::Status::error(
            "Q5_1 Linear dimensions must be non-zero and cols must be a multiple of 32");
    }
    if (request.input_size != request.cols || request.output_size < request.rows) {
        return cpp_llm_engine::core::Status::error("Q5_1 Linear buffer sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ5_1BlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        request.rows * blocks_per_row > std::numeric_limits<std::size_t>::max() / kQ5_1BlockBytes) {
        return cpp_llm_engine::core::Status::error("Q5_1 Linear payload is too large");
    }
    if (request.weight_bytes != request.rows * blocks_per_row * kQ5_1BlockBytes) {
        return cpp_llm_engine::core::Status::error("Q5_1 Linear byte count does not match shape");
    }
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace

bool Q5_1LinearReference::forward(const std::uint8_t* weights, std::size_t weight_bytes,
                                  std::size_t rows, std::size_t cols,
                                  const std::vector<float>& input,
                                  std::vector<float>* output, std::string* error) {
    if (weights == nullptr || output == nullptr) return fail(error, "Q5_1 Linear input and output must not be null");
    std::vector<float> decoded;
    if (!dequantize_q5_1(weights, weight_bytes, rows * cols, &decoded, error)) return false;
    if (rows == 0 || cols == 0 || input.size() != cols) return fail(error, "Q5_1 Linear shape does not match input");
    output->assign(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            (*output)[row] += decoded[row * cols + col] * input[col];
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

cpp_llm_engine::core::BackendKind Q5_1LinearFusedBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q5_1LinearFusedBackend::name() const { return "q5_1-fused-cpu"; }

cpp_llm_engine::core::Status Q5_1LinearFusedBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    const auto status = validate(request);
    if (!status.is_ok()) return status;
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    const std::size_t blocks_per_row = request.cols / kQ5_1BlockElements;
    std::array<float, kQ5_1BlockElements> block_values{};
    for (std::size_t row = 0; row < request.rows; ++row) {
        float sum = 0.0f;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            decode_q5_1_block(weights + (row * blocks_per_row + block) * kQ5_1BlockBytes,
                              block_values.data());
            for (std::size_t value = 0; value < kQ5_1BlockElements; ++value) {
                sum += block_values[value] * request.input[block * kQ5_1BlockElements + value];
            }
        }
        request.output[row] = sum;
    }
    return cpp_llm_engine::core::Status::ok();
}

const Q5_1LinearFusedBackend& q5_1_fused_backend() {
    static const Q5_1LinearFusedBackend backend;
    return backend;
}

}  // namespace chapter05_quantization
