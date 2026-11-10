// Chapter 05：实现 Q2_K 不展开完整权重的 CPU Linear。
#include "chapter05_quantization/q2k_linear.h"

#include "chapter05_quantization/q2k_dequantize.h"

#include <array>
#include <limits>

namespace chapter05_quantization {
namespace {

cpp_llm_engine::core::Status validate(
    const cpp_llm_engine::core::LinearRequest& request) {
    using cpp_llm_engine::core::Status;
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q2K) {
        return Status::error("Q2_K backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr ||
        request.output == nullptr) {
        return Status::error("Q2_K Linear buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 ||
        request.cols % kQ2KBlockElements != 0) {
        return Status::error(
            "Q2_K Linear dimensions must be non-zero and block aligned");
    }
    if (request.input_size != request.cols ||
        request.output_size < request.rows) {
        return Status::error("Q2_K Linear buffer sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ2KBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() /
                           blocks_per_row ||
        request.rows * blocks_per_row >
            std::numeric_limits<std::size_t>::max() / kQ2KBlockBytes ||
        request.weight_bytes !=
            request.rows * blocks_per_row * kQ2KBlockBytes) {
        return Status::error("Q2_K Linear byte count does not match shape");
    }
    return Status::ok();
}

}  // namespace

cpp_llm_engine::core::BackendKind Q2KLinearBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q2KLinearBackend::name() const {
    return "q2k-fused-cpu";
}

cpp_llm_engine::core::Status Q2KLinearBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    const auto status = validate(request);
    if (!status.is_ok()) return status;
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    const std::size_t blocks_per_row = request.cols / kQ2KBlockElements;
    std::array<float, kQ2KBlockElements> values{};
    for (std::size_t row = 0; row < request.rows; ++row) {
        float sum = 0.0F;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            decode_q2_k_block(
                weights + (row * blocks_per_row + block) * kQ2KBlockBytes,
                values.data());
            const float* input =
                request.input + block * kQ2KBlockElements;
            for (std::size_t index = 0; index < kQ2KBlockElements; ++index) {
                sum += values[index] * input[index];
            }
        }
        request.output[row] = sum;
    }
    return cpp_llm_engine::core::Status::ok();
}

const Q2KLinearBackend& q2k_backend() {
    static const Q2KLinearBackend backend;
    return backend;
}

}  // namespace chapter05_quantization
