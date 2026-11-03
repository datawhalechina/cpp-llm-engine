// Chapter 05：量化 Linear backend policy 实现。
// 职责：集中完成 weight format、backend 偏好与可用实现之间的派发。
// 契约：scalar backend 始终可用；AVX2 必须由 registry 显式提供。
// 验证：chapter05_quantization_test.cpp。
#include "chapter05_quantization/quantized_linear_policy.h"

#include "chapter05_quantization/q4k_linear.h"
#include "chapter05_quantization/q6k_linear.h"
#include "chapter05_quantization/q8_0_linear.h"

#include <string>

namespace chapter05_quantization {
namespace {

cpp_llm_engine::core::Status validate_backend(
    QuantizedLinearBackendKind kind,
    const cpp_llm_engine::core::LinearBackend* avx2,
    const char* format) {
    if (kind == QuantizedLinearBackendKind::ScalarFused) {
        return cpp_llm_engine::core::Status::ok();
    }
    if (kind == QuantizedLinearBackendKind::Avx2Fused) {
        if (avx2 != nullptr) return cpp_llm_engine::core::Status::ok();
        return cpp_llm_engine::core::Status::error(
            std::string(format) + " AVX2 backend is unavailable");
    }
    return cpp_llm_engine::core::Status::error(
        std::string(format) + " backend kind is invalid");
}

const cpp_llm_engine::core::LinearBackend* select_backend(
    cpp_llm_engine::core::WeightFormat format,
    const QuantizedLinearPolicy& policy,
    const QuantizedLinearBackendRegistry& registry) {
    if (format == cpp_llm_engine::core::WeightFormat::Q4K) {
        return policy.q4k == QuantizedLinearBackendKind::Avx2Fused
            ? registry.q4k_avx2 : &q4k_fused_backend();
    }
    if (format == cpp_llm_engine::core::WeightFormat::Q6K) {
        return policy.q6k == QuantizedLinearBackendKind::Avx2Fused
            ? registry.q6k_avx2 : &q6k_fused_backend();
    }
    if (format == cpp_llm_engine::core::WeightFormat::Q8_0) {
        return policy.q8_0 == QuantizedLinearBackendKind::Avx2Fused
            ? registry.q8_0_avx2 : &q8_0_fused_backend();
    }
    return nullptr;
}

}  // namespace

cpp_llm_engine::core::Status validate_quantized_linear_policy(
    const QuantizedLinearPolicy& policy,
    const QuantizedLinearBackendRegistry& registry) {
    auto status = validate_backend(policy.q4k, registry.q4k_avx2, "Q4_K");
    if (!status.is_ok()) return status;
    status = validate_backend(policy.q6k, registry.q6k_avx2, "Q6_K");
    if (!status.is_ok()) return status;
    return validate_backend(policy.q8_0, registry.q8_0_avx2, "Q8_0");
}

cpp_llm_engine::core::Status compute_quantized_linear(
    const cpp_llm_engine::core::LinearRequest& request,
    const QuantizedLinearPolicy& policy,
    const QuantizedLinearBackendRegistry& registry) {
    cpp_llm_engine::core::Status selection_status = cpp_llm_engine::core::Status::ok();
    if (request.weight_format == cpp_llm_engine::core::WeightFormat::Q4K) {
        selection_status = validate_backend(policy.q4k, registry.q4k_avx2, "Q4_K");
    } else if (request.weight_format == cpp_llm_engine::core::WeightFormat::Q6K) {
        selection_status = validate_backend(policy.q6k, registry.q6k_avx2, "Q6_K");
    } else if (request.weight_format == cpp_llm_engine::core::WeightFormat::Q8_0) {
        selection_status = validate_backend(policy.q8_0, registry.q8_0_avx2, "Q8_0");
    } else {
        return cpp_llm_engine::core::Status::error(
            "quantized Linear policy only supports Q4_K, Q6_K, and Q8_0");
    }
    if (!selection_status.is_ok()) return selection_status;

    const auto* backend = select_backend(request.weight_format, policy, registry);
    return backend->compute(request);
}

cpp_llm_engine::core::Status compute_quantized_linear_batch(
    const cpp_llm_engine::core::LinearBatchRequest& request,
    const QuantizedLinearPolicy& policy,
    const QuantizedLinearBackendRegistry& registry) {
    cpp_llm_engine::core::Status selection_status = cpp_llm_engine::core::Status::ok();
    if (request.weight_format == cpp_llm_engine::core::WeightFormat::Q4K) {
        selection_status = validate_backend(policy.q4k, registry.q4k_avx2, "Q4_K");
    } else if (request.weight_format == cpp_llm_engine::core::WeightFormat::Q6K) {
        selection_status = validate_backend(policy.q6k, registry.q6k_avx2, "Q6_K");
    } else if (request.weight_format == cpp_llm_engine::core::WeightFormat::Q8_0) {
        selection_status = validate_backend(policy.q8_0, registry.q8_0_avx2, "Q8_0");
    } else {
        return cpp_llm_engine::core::Status::error(
            "quantized Linear batch only supports Q4_K, Q6_K, and Q8_0");
    }
    if (!selection_status.is_ok()) return selection_status;

    const auto* backend = select_backend(request.weight_format, policy, registry);
    return backend->compute_batch(request);
}

}  // namespace chapter05_quantization
