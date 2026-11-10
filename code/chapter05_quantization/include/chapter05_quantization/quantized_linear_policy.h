// Chapter 05：量化 Linear backend 选择策略。
// 职责：按量化格式从统一 policy 选择 scalar 或可选 AVX2 backend。
// 契约：registry 只暴露当前构建和运行机器可用的可选 backend。
// 验证：chapter05_quantization_test.cpp 覆盖选择、执行和缺失能力错误。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

#include <cstdint>

namespace chapter05_quantization {

enum class QuantizedLinearBackendKind : std::uint8_t {
    ScalarFused = 0,
    Avx2Fused,
};

struct QuantizedLinearPolicy final {
    QuantizedLinearBackendKind q4k = QuantizedLinearBackendKind::ScalarFused;
    QuantizedLinearBackendKind q6k = QuantizedLinearBackendKind::ScalarFused;
    QuantizedLinearBackendKind q8_0 = QuantizedLinearBackendKind::ScalarFused;
    QuantizedLinearBackendKind q3k = QuantizedLinearBackendKind::ScalarFused;
    QuantizedLinearBackendKind q5_0 = QuantizedLinearBackendKind::ScalarFused;
    QuantizedLinearBackendKind q5_1 = QuantizedLinearBackendKind::ScalarFused;
    QuantizedLinearBackendKind q5k = QuantizedLinearBackendKind::ScalarFused;
};

struct QuantizedLinearBackendRegistry final {
    const cpp_llm_engine::core::LinearBackend* q4k_avx2 = nullptr;
    const cpp_llm_engine::core::LinearBackend* q6k_avx2 = nullptr;
    const cpp_llm_engine::core::LinearBackend* q8_0_avx2 = nullptr;
    const cpp_llm_engine::core::LinearBackend* q3k_avx2 = nullptr;
    const cpp_llm_engine::core::LinearBackend* q5_0_avx2 = nullptr;
    const cpp_llm_engine::core::LinearBackend* q5_1_avx2 = nullptr;
    const cpp_llm_engine::core::LinearBackend* q5k_avx2 = nullptr;
};

cpp_llm_engine::core::Status validate_quantized_linear_policy(
    const QuantizedLinearPolicy& policy,
    const QuantizedLinearBackendRegistry& registry = {});

cpp_llm_engine::core::Status compute_quantized_linear(
    const cpp_llm_engine::core::LinearRequest& request,
    const QuantizedLinearPolicy& policy,
    const QuantizedLinearBackendRegistry& registry = {});

cpp_llm_engine::core::Status compute_quantized_linear_batch(
    const cpp_llm_engine::core::LinearBatchRequest& request,
    const QuantizedLinearPolicy& policy,
    const QuantizedLinearBackendRegistry& registry = {});

}  // namespace chapter05_quantization
