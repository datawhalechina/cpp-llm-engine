// Chapter 05：实现 Q4_K Linear 的 reference、scalar fused 和 CPU 能力探测。
// backend 先校验统一 LinearRequest，再进入对应数值实现。
// 验证入口：chapter05_quantization_test。
#include "chapter05_quantization/q4k_linear.h"
#include "chapter05_quantization/q4k_kernel.h"
#include "chapter05_quantization/q4k_dequantize.h"

#include <limits>

namespace chapter05_quantization {
namespace {

bool fail(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
    return false;
}

}  // namespace

bool Q4KLinearReference::forward(const std::uint8_t* weights,
                                 std::size_t weight_bytes,
                                 std::size_t rows,
                                 std::size_t cols,
                                 const std::vector<float>& input,
                                 std::vector<float>* output,
                                 std::string* error) {
    if (weights == nullptr || output == nullptr) return fail(error, "Q4_K Linear input and output must not be null");
    if (rows == 0 || cols == 0 || cols % kQ4KBlockElements != 0) {
        return fail(error, "Q4_K Linear dimensions must be non-zero and cols must be a multiple of 256");
    }
    if (input.size() != cols) return fail(error, "Q4_K Linear input width does not match cols");
    if (rows > std::numeric_limits<std::size_t>::max() / cols) {
        return fail(error, "Q4_K Linear dimensions overflow host size");
    }
    const std::size_t elements = rows * cols;
    if (elements / kQ4KBlockElements > std::numeric_limits<std::size_t>::max() / kQ4KBlockBytes) {
        return fail(error, "Q4_K Linear payload is too large");
    }
    const std::size_t expected_bytes = elements / kQ4KBlockElements * kQ4KBlockBytes;
    if (weight_bytes != expected_bytes) return fail(error, "Q4_K Linear byte count does not match shape");

    std::vector<float> decoded;
    if (!dequantize_q4_k(weights, weight_bytes, elements, &decoded, error)) return false;
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

bool Q4KLinearFused::forward(const std::uint8_t* weights,
                             std::size_t weight_bytes,
                             std::size_t rows,
                             std::size_t cols,
                             const std::vector<float>& input,
                             std::vector<float>* output,
                             std::string* error) {
    if (weights == nullptr || output == nullptr) return fail(error, "Q4_K Linear input and output must not be null");
    if (rows == 0 || cols == 0 || cols % kQ4KBlockElements != 0) {
        return fail(error, "Q4_K Linear dimensions must be non-zero and cols must be a multiple of 256");
    }
    if (input.size() != cols) return fail(error, "Q4_K Linear input width does not match cols");
    const std::size_t blocks_per_row = cols / kQ4KBlockElements;
    if (rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        rows * blocks_per_row > std::numeric_limits<std::size_t>::max() / kQ4KBlockBytes) {
        return fail(error, "Q4_K Linear payload is too large");
    }
    const std::size_t expected_bytes = rows * blocks_per_row * kQ4KBlockBytes;
    if (weight_bytes != expected_bytes) return fail(error, "Q4_K Linear byte count does not match shape");
    output->assign(rows, 0.0f);
    const cpp_llm_engine::core::LinearRequest request{
        weights, weight_bytes, input.data(), input.size(), output->data(), output->size(),
        rows, cols, cpp_llm_engine::core::WeightFormat::Q4K};
    const auto status = q4k_fused_backend().compute(request);
    if (!status.is_ok()) return fail(error, status.message().c_str());
    if (error != nullptr) error->clear();
    return true;
}

cpp_llm_engine::core::BackendKind Q4KLinearFusedBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q4KLinearFusedBackend::name() const {
    return "q4k-fused-cpu";
}

cpp_llm_engine::core::Status Q4KLinearFusedBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    return q4k_fused_cpu_kernel(request);
}

cpp_llm_engine::core::Status Q4KLinearFusedBackend::compute_batch(
    const cpp_llm_engine::core::LinearBatchRequest& request) const {
    return q4k_fused_cpu_batch_kernel(request);
}

const Q4KLinearFusedBackend& q4k_fused_backend() {
    static const Q4KLinearFusedBackend backend;
    return backend;
}

bool q4k_avx2_supported() {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

}  // namespace chapter05_quantization
