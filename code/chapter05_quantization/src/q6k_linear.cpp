// Chapter 05：实现 Q6_K Linear 的 reference、fused 入口和 Core backend 适配。
// backend 负责统一请求校验，数值计算委托给对应实现。
// 验证入口：chapter05_quantization_test。
#include "chapter05_quantization/q6k_linear.h"

#include <limits>

#include "chapter05_quantization/q6k_dequantize.h"
#include "chapter05_quantization/q6k_kernel.h"

namespace chapter05_quantization {
namespace {

bool fail(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
    return false;
}

bool validate(const std::uint8_t* weights,
              std::size_t weight_bytes,
              std::size_t rows,
              std::size_t cols,
              const std::vector<float>& input,
              std::string* error) {
    if (weights == nullptr) return fail(error, "Q6_K Linear weights must not be null");
    if (rows == 0 || cols == 0 || cols % kQ6KBlockElements != 0) {
        return fail(error, "Q6_K Linear dimensions must be non-zero and cols must be a multiple of 256");
    }
    if (input.size() != cols) return fail(error, "Q6_K Linear input width does not match cols");
    if (rows > std::numeric_limits<std::size_t>::max() / cols) {
        return fail(error, "Q6_K Linear dimensions overflow host size");
    }
    const std::size_t elements = rows * cols;
    if (elements / kQ6KBlockElements > std::numeric_limits<std::size_t>::max() / kQ6KBlockBytes) {
        return fail(error, "Q6_K Linear payload is too large");
    }
    if (weight_bytes != elements / kQ6KBlockElements * kQ6KBlockBytes) {
        return fail(error, "Q6_K Linear byte count does not match shape");
    }
    return true;
}

}  // namespace

bool Q6KLinearReference::forward(const std::uint8_t* weights,
                                 std::size_t weight_bytes,
                                 std::size_t rows,
                                 std::size_t cols,
                                 const std::vector<float>& input,
                                 std::vector<float>* output,
                                 std::string* error) {
    if (output == nullptr) return fail(error, "Q6_K Linear output must not be null");
    if (!validate(weights, weight_bytes, rows, cols, input, error)) return false;
    std::vector<float> decoded;
    if (!dequantize_q6_k(weights, weight_bytes, rows * cols, &decoded, error)) return false;
    output->assign(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            (*output)[row] += decoded[row * cols + col] * input[col];
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

bool Q6KLinearFused::forward(const std::uint8_t* weights,
                             std::size_t weight_bytes,
                             std::size_t rows,
                             std::size_t cols,
                             const std::vector<float>& input,
                             std::vector<float>* output,
                             std::string* error) {
    if (output == nullptr) return fail(error, "Q6_K Linear output must not be null");
    if (!validate(weights, weight_bytes, rows, cols, input, error)) return false;
    output->assign(rows, 0.0f);
    const cpp_llm_engine::core::LinearRequest request{
        weights, weight_bytes, input.data(), input.size(), output->data(), output->size(),
        rows, cols, cpp_llm_engine::core::WeightFormat::Q6K};
    const auto status = q6k_fused_backend().compute(request);
    if (!status.is_ok()) return fail(error, status.message().c_str());
    if (error != nullptr) error->clear();
    return true;
}

cpp_llm_engine::core::BackendKind Q6KLinearFusedBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q6KLinearFusedBackend::name() const {
    return "q6k-fused-cpu";
}

cpp_llm_engine::core::Status Q6KLinearFusedBackend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    return q6k_fused_cpu_kernel(request);
}

cpp_llm_engine::core::Status Q6KLinearFusedBackend::compute_batch(
    const cpp_llm_engine::core::LinearBatchRequest& request) const {
    return q6k_fused_cpu_batch_kernel(request);
}

const Q6KLinearFusedBackend& q6k_fused_backend() {
    static const Q6KLinearFusedBackend backend;
    return backend;
}

bool q6k_avx2_supported() {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

}  // namespace chapter05_quantization
