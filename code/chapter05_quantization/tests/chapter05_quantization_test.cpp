// Chapter 05：验证 INT8、Q4_K、Q6_K、Q8_0 Linear 及 backend policy。
// 覆盖：数值对照、输入错误、能力检查和统一策略分派。
#include "chapter05_quantization/dequantize_op.h"
#include "chapter05_quantization/linear_layer.h"
#include "chapter05_quantization/quantized_linear.h"
#include "chapter05_quantization/quantized_linear_policy.h"
#include "chapter05_quantization/q4k_dequantize.h"
#include "chapter05_quantization/q4k_kernel.h"
#include "chapter05_quantization/q4k_linear.h"
#include "chapter05_quantization/q6k_dequantize.h"
#include "chapter05_quantization/q6k_linear.h"
#include "chapter05_quantization/q8_0_dequantize.h"
#include "chapter05_quantization/q8_0_linear.h"
#include "cpp_llm_engine/testing/test.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

int main() {
    using namespace chapter05_quantization;

    cpp_llm_engine::testing::TestSuite suite("chapter05_quantization");
    suite.run("量化格式、后端策略与数值参考保持一致", [&] {
    const Tensor quantized = Tensor::from_int8(2, 2, {1, 2, 3, 4}, 0.1f);
    CPP_LLM_REQUIRE(suite, quantized.ready());
    CPP_LLM_CHECK(suite, quantized.is_int8());
    CPP_LLM_CHECK(suite, std::abs(quantized.at(0, 0) - 0.1f) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(quantized.at(1, 1) - 0.4f) < 1e-5f);

    const DequantizeOp dequantize_op;
    const Tensor dequantized = dequantize_op.forward(quantized);
    CPP_LLM_REQUIRE(suite, dequantized.ready());
    CPP_LLM_CHECK(suite, dequantized.is_float());
    CPP_LLM_CHECK(suite, std::abs(dequantized.at(0, 0) - 0.1f) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(dequantized.at(1, 1) - 0.4f) < 1e-5f);

    const Tensor input = Tensor::from_float(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    const LinearLayer reference_linear(2, 2);
    const QuantizedLinear quantized_linear(2, 2, 0.1f);

    const Tensor reference_output = reference_linear.forward(input);
    const Tensor quantized_output = quantized_linear.forward(input);

    CPP_LLM_REQUIRE(suite, reference_output.ready());
    CPP_LLM_REQUIRE(suite, quantized_output.ready());
    CPP_LLM_CHECK(suite, reference_output.rows() == quantized_output.rows());
    CPP_LLM_CHECK(suite, reference_output.cols() == quantized_output.cols());
    CPP_LLM_CHECK(suite, std::abs(reference_output.at(0, 0) - quantized_output.at(0, 0)) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(reference_output.at(1, 1) - quantized_output.at(1, 1)) < 1e-5f);

    std::vector<std::uint8_t> q8_0(kQ8_0BlockBytes, 0);
    q8_0[0] = 0x00;
    q8_0[1] = 0x38;  // little-endian half 0.5
    for (std::size_t i = 0; i < kQ8_0BlockElements; ++i) {
        q8_0[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(
            static_cast<int>(i) - 16));
    }
    std::vector<float> q8_0_output;
    std::string error;
    CPP_LLM_REQUIRE(suite, dequantize_q8_0(q8_0.data(), q8_0.size(), 32, &q8_0_output, &error));
    CPP_LLM_CHECK(suite, error.empty());
    CPP_LLM_REQUIRE(suite, q8_0_output.size() == 32);
    CPP_LLM_CHECK(suite, std::abs(q8_0_output.front() - (-8.0f)) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(q8_0_output[16]) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(q8_0_output.back() - 7.5f) < 1e-5f);
    CPP_LLM_CHECK(suite, !dequantize_q8_0(q8_0.data(), q8_0.size() - 1, 32, &q8_0_output, &error));
    CPP_LLM_CHECK(suite, error == "Q8_0 byte count does not match block count");

    std::vector<std::uint8_t> q8_0_matrix(kQ8_0BlockBytes * 2, 0);
    std::copy(q8_0.begin(), q8_0.end(), q8_0_matrix.begin());
    std::copy(q8_0.begin(), q8_0.end(), q8_0_matrix.begin() + kQ8_0BlockBytes);
    q8_0_matrix[kQ8_0BlockBytes] = 0x00;
    q8_0_matrix[kQ8_0BlockBytes + 1] = 0x3c;  // little-endian half 1.0
    const std::vector<float> q8_0_input(kQ8_0BlockElements, 2.0f);
    std::vector<float> q8_0_reference;
    std::vector<float> q8_0_fused;
    CPP_LLM_REQUIRE(suite, Q8_0LinearReference::forward(
        q8_0_matrix.data(), q8_0_matrix.size(), 2, 32,
        q8_0_input, &q8_0_reference, &error));
    CPP_LLM_REQUIRE(suite, Q8_0LinearFused::forward(
        q8_0_matrix.data(), q8_0_matrix.size(), 2, 32,
        q8_0_input, &q8_0_fused, &error));
    CPP_LLM_REQUIRE(suite, q8_0_reference.size() == 2 && q8_0_fused.size() == 2);
    CPP_LLM_CHECK(suite, std::abs(q8_0_reference[0] - (-16.0f)) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(q8_0_reference[1] - (-32.0f)) < 1e-5f);
    CPP_LLM_CHECK(suite, q8_0_fused == q8_0_reference);
    CPP_LLM_CHECK(suite, q8_0_fused_backend().kind() == cpp_llm_engine::core::BackendKind::CPU);
    CPP_LLM_CHECK(suite, q8_0_fused_backend().name() == "q8_0-fused-cpu");
    std::vector<float> q8_0_policy_output(2, 0.0f);
    const cpp_llm_engine::core::LinearRequest q8_0_request{
        q8_0_matrix.data(), q8_0_matrix.size(),
        q8_0_input.data(), q8_0_input.size(),
        q8_0_policy_output.data(), q8_0_policy_output.size(),
        2, 32, cpp_llm_engine::core::WeightFormat::Q8_0};
    const QuantizedLinearPolicy q8_0_scalar_policy;
    CPP_LLM_REQUIRE(suite, compute_quantized_linear(q8_0_request, q8_0_scalar_policy).is_ok());
    CPP_LLM_CHECK(suite, q8_0_policy_output == q8_0_reference);
    const QuantizedLinearPolicy missing_q8_avx2{
        QuantizedLinearBackendKind::ScalarFused,
        QuantizedLinearBackendKind::ScalarFused,
        QuantizedLinearBackendKind::Avx2Fused};
    const auto missing_q8_status = validate_quantized_linear_policy(missing_q8_avx2);
    CPP_LLM_CHECK(suite, !missing_q8_status.is_ok());
    CPP_LLM_CHECK(suite, missing_q8_status.message() == "Q8_0 AVX2 backend is unavailable");
    CPP_LLM_CHECK(suite, !Q8_0LinearFused::forward(
        q8_0_matrix.data(), q8_0_matrix.size() - 1, 2, 32,
        q8_0_input, &q8_0_fused, &error));
    CPP_LLM_CHECK(suite, error == "Q8_0 Linear byte count does not match shape");

    std::vector<std::uint8_t> q4k(144, 0);
    q4k[0] = 0x00;
    q4k[1] = 0x3c; // little-endian half 1.0
    for (std::size_t i = 4; i < 12; ++i) q4k[i] = 1;
    for (std::size_t i = 12; i < q4k.size(); ++i) q4k[i] = 0x11;
    std::vector<float> q4k_output;
    CPP_LLM_REQUIRE(suite, dequantize_q4_k(q4k.data(), q4k.size(), 256, &q4k_output, &error));
    CPP_LLM_CHECK(suite, error.empty());
    CPP_LLM_REQUIRE(suite, q4k_output.size() == 256);
    for (const float value : q4k_output) {
        CPP_LLM_CHECK(suite, std::abs(value - 1.0f) < 1e-5f);
    }

    std::vector<float> linear_output;
    const std::vector<float> linear_input(256, 2.0f);
    CPP_LLM_REQUIRE(suite, Q4KLinearReference::forward(
        q4k.data(), q4k.size(), 1, 256, linear_input, &linear_output, &error));
    CPP_LLM_REQUIRE(suite, linear_output.size() == 1);
    CPP_LLM_CHECK(suite, std::abs(linear_output[0] - 512.0f) < 1e-3f);

    std::vector<float> fused_output;
    CPP_LLM_REQUIRE(suite, Q4KLinearFused::forward(
        q4k.data(), q4k.size(), 1, 256, linear_input, &fused_output, &error));
    CPP_LLM_REQUIRE(suite, fused_output.size() == linear_output.size());
    CPP_LLM_CHECK(suite, std::abs(fused_output[0] - linear_output[0]) < 1e-3f);
    CPP_LLM_CHECK(suite, q4k_fused_backend().kind() == cpp_llm_engine::core::BackendKind::CPU);
    CPP_LLM_CHECK(suite, q4k_fused_backend().name() == "q4k-fused-cpu");
    std::vector<float> contract_output(1, 0.0f);
    const cpp_llm_engine::core::LinearRequest request{
        q4k.data(), q4k.size(), linear_input.data(), linear_input.size(),
        contract_output.data(), contract_output.size(), 1, 256,
        cpp_llm_engine::core::WeightFormat::Q4K};
    CPP_LLM_REQUIRE(suite, q4k_fused_backend().compute(request).is_ok());
    CPP_LLM_CHECK(suite, std::abs(contract_output[0] - linear_output[0]) < 1e-3f);
    const QuantizedLinearPolicy scalar_policy;
    contract_output[0] = 0.0f;
    CPP_LLM_REQUIRE(suite, compute_quantized_linear(request, scalar_policy).is_ok());
    CPP_LLM_CHECK(suite, std::abs(contract_output[0] - linear_output[0]) < 1e-3f);
    const QuantizedLinearPolicy missing_q4_avx2{
        QuantizedLinearBackendKind::Avx2Fused,
        QuantizedLinearBackendKind::ScalarFused};
    const auto missing_q4_status = validate_quantized_linear_policy(missing_q4_avx2);
    CPP_LLM_CHECK(suite, !missing_q4_status.is_ok());
    CPP_LLM_CHECK(suite, missing_q4_status.message() == "Q4_K AVX2 backend is unavailable");
    const QuantizedLinearPolicy invalid_q4_policy{
        static_cast<QuantizedLinearBackendKind>(255),
        QuantizedLinearBackendKind::ScalarFused};
    const auto invalid_q4_status = compute_quantized_linear(request, invalid_q4_policy);
    CPP_LLM_CHECK(suite, !invalid_q4_status.is_ok());
    CPP_LLM_CHECK(suite, invalid_q4_status.message() == "Q4_K backend kind is invalid");
#ifdef CHAPTER05_HAS_AVX2
    if (q4k_avx2_supported()) {
        std::vector<float> q4k_avx2_output;
        CPP_LLM_REQUIRE(suite, Q4KLinearAvx2::forward(
            q4k.data(), q4k.size(), 1, 256, linear_input, &q4k_avx2_output, &error));
        CPP_LLM_REQUIRE(suite, q4k_avx2_output.size() == linear_output.size());
        CPP_LLM_CHECK(suite, std::abs(q4k_avx2_output[0] - linear_output[0]) < 1e-3f);
        CPP_LLM_CHECK(suite, q4k_avx2_backend().kind() == cpp_llm_engine::core::BackendKind::CPU);
        CPP_LLM_CHECK(suite, q4k_avx2_backend().name() == "q4k-fused-avx2");
        const QuantizedLinearBackendRegistry q4k_registry{&q4k_avx2_backend(), nullptr};
        CPP_LLM_REQUIRE(suite, validate_quantized_linear_policy(missing_q4_avx2, q4k_registry).is_ok());
        contract_output[0] = 0.0f;
        CPP_LLM_CHECK(suite, compute_quantized_linear(
            request, missing_q4_avx2, q4k_registry).is_ok());
        CPP_LLM_CHECK(suite, std::abs(contract_output[0] - linear_output[0]) < 1e-3f);

        std::vector<std::uint8_t> varied_q4k(kQ4KBlockBytes, 0);
        varied_q4k[0] = 0x00;
        varied_q4k[1] = 0x38;  // little-endian half 0.5
        varied_q4k[2] = 0x00;
        varied_q4k[3] = 0x34;  // little-endian half 0.25
        for (std::size_t i = 0; i < 12; ++i) {
            varied_q4k[4 + i] = static_cast<std::uint8_t>(3 + i * 7);
        }
        for (std::size_t i = 0; i < 128; ++i) {
            varied_q4k[16 + i] = static_cast<std::uint8_t>((i * 29 + 11) & 0xff);
        }
        std::vector<float> varied_input(256, 0.0f);
        for (std::size_t i = 0; i < varied_input.size(); ++i) {
            varied_input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 7.0f;
        }
        std::vector<float> varied_reference;
        std::vector<float> varied_fused;
        std::vector<float> varied_avx2;
        CPP_LLM_REQUIRE(suite, Q4KLinearReference::forward(
            varied_q4k.data(), varied_q4k.size(), 1, 256, varied_input,
            &varied_reference, &error));
        CPP_LLM_REQUIRE(suite, Q4KLinearFused::forward(
            varied_q4k.data(), varied_q4k.size(), 1, 256, varied_input,
            &varied_fused, &error));
        CPP_LLM_REQUIRE(suite, Q4KLinearAvx2::forward(
            varied_q4k.data(), varied_q4k.size(), 1, 256, varied_input,
            &varied_avx2, &error));
        CPP_LLM_CHECK(suite, std::abs(varied_fused[0] - varied_reference[0]) < 1e-2f);
        CPP_LLM_CHECK(suite, std::abs(varied_avx2[0] - varied_reference[0]) < 1e-2f);
    }
#endif

    std::vector<std::uint8_t> two_block_q4k(q4k.size() * 2, 0);
    std::copy(q4k.begin(), q4k.end(), two_block_q4k.begin());
    std::copy(q4k.begin(), q4k.end(), two_block_q4k.begin() + q4k.size());
    two_block_q4k[q4k.size() + 1] = 0x40; // little-endian half 2.0
    std::vector<float> two_block_input(512, 1.0f);
    std::vector<float> two_block_reference;
    std::vector<float> two_block_fused;
    CPP_LLM_REQUIRE(suite, Q4KLinearReference::forward(
        two_block_q4k.data(), two_block_q4k.size(), 1, 512,
        two_block_input, &two_block_reference, &error));
    CPP_LLM_REQUIRE(suite, Q4KLinearFused::forward(
        two_block_q4k.data(), two_block_q4k.size(), 1, 512,
        two_block_input, &two_block_fused, &error));
    CPP_LLM_CHECK(suite, std::abs(two_block_reference[0] - 768.0f) < 1e-3f);
    CPP_LLM_CHECK(suite, std::abs(two_block_fused[0] - two_block_reference[0]) < 1e-3f);

    CPP_LLM_CHECK(suite, !dequantize_q4_k(q4k.data(), q4k.size() - 1, 256, &q4k_output, &error));
    CPP_LLM_CHECK(suite, error == "Q4_K byte count does not match block count");

    std::vector<std::uint8_t> q6k(210, 0);
    q6k[0] = 0x21;   // q1=1, q3=2 before upper two bits are added.
    q6k[32] = 0x43;  // q2=3, q4=4 before upper two bits are added.
    q6k[128] = 0x39;  // q1/q2/q3/q4 upper bits: 1, 2, 3, 0.
    for (std::size_t i = 192; i < 208; ++i) q6k[i] = 1;
    q6k[208] = 0x00;
    q6k[209] = 0x3c;  // little-endian half 1.0
    std::vector<float> q6k_output;
    CPP_LLM_REQUIRE(suite, dequantize_q6_k(q6k.data(), q6k.size(), 256, &q6k_output, &error));
    CPP_LLM_CHECK(suite, error.empty());
    CPP_LLM_REQUIRE(suite, q6k_output.size() == 256);
    CPP_LLM_CHECK(suite, std::abs(q6k_output[0] - (-15.0f)) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(q6k_output[32] - 3.0f) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(q6k_output[64] - 18.0f) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(q6k_output[96] - (-28.0f)) < 1e-5f);
    CPP_LLM_CHECK(suite, std::abs(q6k_output[127] - (-32.0f)) < 1e-5f);
    CPP_LLM_CHECK(suite, !dequantize_q6_k(q6k.data(), q6k.size() - 1, 256, &q6k_output, &error));
    CPP_LLM_CHECK(suite, error == "Q6_K byte count does not match block count");

    const std::vector<float> q6k_input(256, 1.0f);
    std::vector<float> q6k_linear_reference;
    std::vector<float> q6k_linear_fused;
    CPP_LLM_REQUIRE(suite, Q6KLinearReference::forward(
        q6k.data(), q6k.size(), 1, 256, q6k_input, &q6k_linear_reference, &error));
    CPP_LLM_REQUIRE(suite, Q6KLinearFused::forward(
        q6k.data(), q6k.size(), 1, 256, q6k_input, &q6k_linear_fused, &error));
    CPP_LLM_REQUIRE(suite, q6k_linear_reference.size() == 1);
    CPP_LLM_REQUIRE(suite, q6k_linear_fused.size() == q6k_linear_reference.size());
    CPP_LLM_CHECK(suite, std::abs(q6k_linear_reference[0] - (-8086.0f)) < 1e-3f);
    CPP_LLM_CHECK(suite, std::abs(q6k_linear_fused[0] - q6k_linear_reference[0]) < 1e-3f);
    CPP_LLM_CHECK(suite, q6k_fused_backend().kind() == cpp_llm_engine::core::BackendKind::CPU);
    CPP_LLM_CHECK(suite, q6k_fused_backend().name() == "q6k-fused-cpu");
    std::vector<float> q6k_backend_output(1, 0.0f);
    const cpp_llm_engine::core::LinearRequest q6k_request{
        q6k.data(), q6k.size(), q6k_input.data(), q6k_input.size(),
        q6k_backend_output.data(), q6k_backend_output.size(), 1, 256,
        cpp_llm_engine::core::WeightFormat::Q6K};
    CPP_LLM_REQUIRE(suite, q6k_fused_backend().compute(q6k_request).is_ok());
    CPP_LLM_CHECK(suite, std::abs(q6k_backend_output[0] - q6k_linear_reference[0]) < 1e-3f);
    q6k_backend_output[0] = 0.0f;
    CPP_LLM_REQUIRE(suite, compute_quantized_linear(q6k_request, scalar_policy).is_ok());
    CPP_LLM_CHECK(suite, std::abs(q6k_backend_output[0] - q6k_linear_reference[0]) < 1e-3f);
#ifdef CHAPTER05_HAS_AVX2
    if (q6k_avx2_supported()) {
        std::vector<float> q6k_avx2_output;
        CPP_LLM_REQUIRE(suite, Q6KLinearAvx2::forward(
            q6k.data(), q6k.size(), 1, 256, q6k_input, &q6k_avx2_output, &error));
        CPP_LLM_REQUIRE(suite, q6k_avx2_output.size() == q6k_linear_reference.size());
        CPP_LLM_CHECK(suite, std::abs(q6k_avx2_output[0] - q6k_linear_reference[0]) < 1e-3f);
        CPP_LLM_CHECK(suite, q6k_avx2_backend().kind() == cpp_llm_engine::core::BackendKind::CPU);
        CPP_LLM_CHECK(suite, q6k_avx2_backend().name() == "q6k-fused-avx2");
        const QuantizedLinearPolicy q6k_avx2_policy{
            QuantizedLinearBackendKind::ScalarFused,
            QuantizedLinearBackendKind::Avx2Fused};
        const QuantizedLinearBackendRegistry q6k_registry{nullptr, &q6k_avx2_backend()};
        CPP_LLM_REQUIRE(suite, validate_quantized_linear_policy(q6k_avx2_policy, q6k_registry).is_ok());
        q6k_backend_output[0] = 0.0f;
        CPP_LLM_CHECK(suite, compute_quantized_linear(
            q6k_request, q6k_avx2_policy, q6k_registry).is_ok());
        CPP_LLM_CHECK(suite, std::abs(q6k_backend_output[0] - q6k_linear_reference[0]) < 1e-3f);
    }
#endif

    const cpp_llm_engine::core::LinearRequest wrong_format_request{
        q4k.data(), q4k.size(), linear_input.data(), linear_input.size(),
        contract_output.data(), contract_output.size(), 1, 256,
        cpp_llm_engine::core::WeightFormat::Float32};
    CPP_LLM_CHECK(suite, !q4k_fused_backend().compute(wrong_format_request).is_ok());
    const auto unsupported_policy_status = compute_quantized_linear(
        wrong_format_request, scalar_policy);
    CPP_LLM_CHECK(suite, !unsupported_policy_status.is_ok());
    CPP_LLM_CHECK(suite, unsupported_policy_status.message() ==
           "quantized Linear policy only supports Q4_K, Q6_K, and Q8_0");

    std::vector<float> rejected_output;
    CPP_LLM_CHECK(suite, !Q4KLinearFused::forward(
        q4k.data(), q4k.size(), std::numeric_limits<std::size_t>::max(), 256,
        linear_input, &rejected_output, &error));
    CPP_LLM_CHECK(suite, error == "Q4_K Linear payload is too large");

    });
    return suite.finish();
}
