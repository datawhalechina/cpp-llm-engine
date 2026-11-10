// Chapter 05：实现 IQ4_NL 不展开完整权重的 CPU Linear。
#include "chapter05_quantization/iq4nl_linear.h"

#include "chapter05_quantization/iq4nl_dequantize.h"

#include <array>
#include <limits>

namespace chapter05_quantization {
namespace {

cpp_llm_engine::core::Status validate(
    const cpp_llm_engine::core::LinearRequest& request) {
    using cpp_llm_engine::core::Status;
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::IQ4_NL) {
        return Status::error(
            "IQ4_NL backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr ||
        request.output == nullptr) {
        return Status::error("IQ4_NL Linear buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 ||
        request.cols % kIQ4NLBlockElements != 0) {
        return Status::error(
            "IQ4_NL Linear dimensions must be non-zero and block aligned");
    }
    if (request.input_size != request.cols ||
        request.output_size < request.rows) {
        return Status::error("IQ4_NL Linear buffer sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kIQ4NLBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() /
                           blocks_per_row ||
        request.rows * blocks_per_row >
            std::numeric_limits<std::size_t>::max() / kIQ4NLBlockBytes ||
        request.weight_bytes !=
            request.rows * blocks_per_row * kIQ4NLBlockBytes) {
        return Status::error("IQ4_NL Linear byte count does not match shape");
    }
    return Status::ok();
}

}  // namespace

cpp_llm_engine::core::BackendKind IQ4NLLinearBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view IQ4NLLinearBackend::name() const {
    return "iq4-nl-fused-cpu";
}

cpp_llm_engine::core::Status IQ4NLLinearBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    const auto status = validate(request);
    if (!status.is_ok()) return status;
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    const std::size_t blocks_per_row = request.cols / kIQ4NLBlockElements;
    std::array<float, kIQ4NLBlockElements> values{};
    for (std::size_t row = 0; row < request.rows; ++row) {
        float sum = 0.0F;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            decode_iq4_nl_block(
                weights + (row * blocks_per_row + block) * kIQ4NLBlockBytes,
                values.data());
            const float* input =
                request.input + block * kIQ4NLBlockElements;
            for (std::size_t index = 0;
                 index < kIQ4NLBlockElements; ++index) {
                sum += values[index] * input[index];
            }
        }
        request.output[row] = sum;
    }
    return cpp_llm_engine::core::Status::ok();
}

const IQ4NLLinearBackend& iq4_nl_backend() {
    static const IQ4NLLinearBackend backend;
    return backend;
}

}  // namespace chapter05_quantization
