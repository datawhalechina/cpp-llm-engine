// Chapter 05：实现 Q5_0 边解码边点积的 CPU Linear。
#include "chapter05_quantization/q5_0_linear.h"

#include "chapter05_quantization/q5_0_dequantize.h"

#include <array>
#include <limits>

namespace chapter05_quantization {

cpp_llm_engine::core::BackendKind Q5_0LinearFusedBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q5_0LinearFusedBackend::name() const {
    return "q5_0-fused-cpu";
}

cpp_llm_engine::core::Status Q5_0LinearFusedBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    using cpp_llm_engine::core::Status;
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q5_0 ||
        request.weights == nullptr || request.input == nullptr ||
        request.output == nullptr || request.rows == 0 || request.cols == 0 ||
        request.cols % kQ5_0BlockElements != 0 ||
        request.input_size != request.cols || request.output_size < request.rows) {
        return Status::error("Q5_0 Linear request is inconsistent");
    }
    const std::size_t blocks_per_row = request.cols / kQ5_0BlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        request.rows * blocks_per_row >
            std::numeric_limits<std::size_t>::max() / kQ5_0BlockBytes ||
        request.weight_bytes !=
            request.rows * blocks_per_row * kQ5_0BlockBytes) {
        return Status::error("Q5_0 Linear payload does not match shape");
    }
    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    std::array<float, kQ5_0BlockElements> decoded{};
    for (std::size_t row = 0; row < request.rows; ++row) {
        float sum = 0.0F;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            decode_q5_0_block(
                weights + (row * blocks_per_row + block) * kQ5_0BlockBytes,
                decoded.data());
            for (std::size_t value = 0; value < kQ5_0BlockElements; ++value) {
                sum += decoded[value] *
                    request.input[block * kQ5_0BlockElements + value];
            }
        }
        request.output[row] = sum;
    }
    return Status::ok();
}

const Q5_0LinearFusedBackend& q5_0_fused_backend() {
    static const Q5_0LinearFusedBackend backend;
    return backend;
}

}  // namespace chapter05_quantization
