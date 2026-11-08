// Chapter 05：验证 INT8、Q4_K、Q6_K、Q8_0 Linear 及 backend policy。
// 每个用例只覆盖一种格式的一项可命名行为，便于定位解码、执行或策略错误。
// 固定 payload 提供独立期望值，可用的 AVX2 实现再与 scalar reference 对照。
// 验证入口：bash code/chapter05_quantization/scripts/run.sh。
#include "chapter05_quantization/dequantize_op.h"
#include "chapter05_quantization/linear_layer.h"
#include "chapter05_quantization/quantized_linear.h"
#include "chapter05_quantization/quantized_linear_policy.h"
#include "chapter05_quantization/q4k_dequantize.h"
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

namespace {

std::vector<std::uint8_t> make_q8_0_block() {
    std::vector<std::uint8_t> block(chapter05_quantization::kQ8_0BlockBytes, 0);
    block[0] = 0x00;
    block[1] = 0x38;  // little-endian half 0.5
    for (std::size_t i = 0; i < chapter05_quantization::kQ8_0BlockElements; ++i) {
        block[2 + i] = static_cast<std::uint8_t>(
            static_cast<std::int8_t>(static_cast<int>(i) - 16));
    }
    return block;
}

std::vector<std::uint8_t> make_unit_q4k_block() {
    std::vector<std::uint8_t> block(chapter05_quantization::kQ4KBlockBytes, 0);
    block[0] = 0x00;
    block[1] = 0x3c;  // little-endian half 1.0
    for (std::size_t i = 4; i < 12; ++i) block[i] = 1;
    for (std::size_t i = 12; i < block.size(); ++i) block[i] = 0x11;
    return block;
}

std::vector<std::uint8_t> make_q6k_block() {
    std::vector<std::uint8_t> block(chapter05_quantization::kQ6KBlockBytes, 0);
    block[0] = 0x21;
    block[32] = 0x43;
    block[128] = 0x39;
    for (std::size_t i = 192; i < 208; ++i) block[i] = 1;
    block[208] = 0x00;
    block[209] = 0x3c;  // little-endian half 1.0
    return block;
}

bool near(float lhs, float rhs, float tolerance = 1e-3f) {
    return std::abs(lhs - rhs) < tolerance;
}

}  // namespace

int main() {
    using namespace chapter05_quantization;

    cpp_llm_engine::testing::TestSuite suite("chapter05_quantization");
    suite.run("Tensor rejects invalid dimensions and quantization scales", [&] {
        const auto max_size = std::numeric_limits<std::size_t>::max();
        CPP_LLM_CHECK(suite, !Tensor::from_float(0, 1, {}).ready());
        CPP_LLM_CHECK(suite, !Tensor::from_float(max_size, 2, {}).ready());
        CPP_LLM_CHECK(suite, !Tensor::from_int8(1, 1, {1}, 0.0f).ready());
        CPP_LLM_CHECK(suite, !Tensor::from_int8(
            1, 1, {1}, std::numeric_limits<float>::infinity()).ready());
        CPP_LLM_CHECK(suite, !Tensor::from_int8(
            1, 1, {1}, std::numeric_limits<float>::quiet_NaN()).ready());
    });

    suite.run("INT8 teaching path matches Float32 reference", [&] {
        const Tensor quantized = Tensor::from_int8(2, 2, {1, 2, 3, 4}, 0.1f);
        CPP_LLM_REQUIRE(suite, quantized.ready());
        CPP_LLM_CHECK(suite, quantized.is_int8());
        CPP_LLM_CHECK(suite, near(quantized.at(0, 0), 0.1f, 1e-5f));
        CPP_LLM_CHECK(suite, near(quantized.at(1, 1), 0.4f, 1e-5f));
        const Tensor dequantized = DequantizeOp{}.forward(quantized);
        CPP_LLM_REQUIRE(suite, dequantized.ready());
        CPP_LLM_CHECK(suite, near(dequantized.at(0, 0), 0.1f, 1e-5f));

        const Tensor input = Tensor::from_float(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
        const Tensor reference_output = LinearLayer(2, 2).forward(input);
        const Tensor quantized_output = QuantizedLinear(2, 2, 0.1f).forward(input);
        CPP_LLM_REQUIRE(suite, reference_output.ready());
        CPP_LLM_REQUIRE(suite, quantized_output.ready());
        CPP_LLM_CHECK(suite, reference_output.same_shape(quantized_output));
        CPP_LLM_CHECK(suite, near(reference_output.at(0, 0), quantized_output.at(0, 0), 1e-5f));
        CPP_LLM_CHECK(suite, near(reference_output.at(1, 1), quantized_output.at(1, 1), 1e-5f));
    });

    suite.run("Q8_0 block decoding matches fixed values", [&] {
        const auto block = make_q8_0_block();
        std::vector<float> output;
        std::string error;
        CPP_LLM_REQUIRE(suite, dequantize_q8_0(
            block.data(), block.size(), kQ8_0BlockElements, &output, &error));
        CPP_LLM_REQUIRE(suite, output.size() == kQ8_0BlockElements);
        CPP_LLM_CHECK(suite, near(output.front(), -8.0f, 1e-5f));
        CPP_LLM_CHECK(suite, near(output[16], 0.0f, 1e-5f));
        CPP_LLM_CHECK(suite, near(output.back(), 7.5f, 1e-5f));
    });

    suite.run("Q8_0 fused Linear and scalar policy match reference", [&] {
        const auto block = make_q8_0_block();
        std::vector<std::uint8_t> weights(kQ8_0BlockBytes * 2, 0);
        std::copy(block.begin(), block.end(), weights.begin());
        std::copy(block.begin(), block.end(), weights.begin() + kQ8_0BlockBytes);
        weights[kQ8_0BlockBytes] = 0x00;
        weights[kQ8_0BlockBytes + 1] = 0x3c;
        const std::vector<float> input(kQ8_0BlockElements, 2.0f);
        std::vector<float> reference;
        std::vector<float> fused;
        std::string error;
        CPP_LLM_REQUIRE(suite, Q8_0LinearReference::forward(
            weights.data(), weights.size(), 2, 32, input, &reference, &error));
        CPP_LLM_REQUIRE(suite, Q8_0LinearFused::forward(
            weights.data(), weights.size(), 2, 32, input, &fused, &error));
        CPP_LLM_CHECK(suite, reference == fused);
        CPP_LLM_CHECK(suite, near(reference[0], -16.0f, 1e-5f));
        CPP_LLM_CHECK(suite, near(reference[1], -32.0f, 1e-5f));

        std::vector<float> output(2, 0.0f);
        const cpp_llm_engine::core::LinearRequest request{
            weights.data(), weights.size(), input.data(), input.size(),
            output.data(), output.size(), 2, 32,
            cpp_llm_engine::core::WeightFormat::Q8_0};
        CPP_LLM_REQUIRE(suite, compute_quantized_linear(request, {}).is_ok());
        CPP_LLM_CHECK(suite, output == reference);
    });

    suite.run("Q8_0 rejects malformed payload and unavailable AVX2", [&] {
        const auto block = make_q8_0_block();
        std::vector<float> output;
        std::string error;
        CPP_LLM_CHECK(suite, !dequantize_q8_0(
            block.data(), block.size() - 1, 32, &output, &error));
        CPP_LLM_CHECK(suite, error == "Q8_0 byte count does not match block count");
        const QuantizedLinearPolicy missing_avx2{
            QuantizedLinearBackendKind::ScalarFused,
            QuantizedLinearBackendKind::ScalarFused,
            QuantizedLinearBackendKind::Avx2Fused};
        const auto status = validate_quantized_linear_policy(missing_avx2);
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, status.message() == "Q8_0 AVX2 backend is unavailable");
    });

    suite.run("Q4_K block decoding and fused Linear match reference", [&] {
        const auto weights = make_unit_q4k_block();
        std::vector<float> decoded;
        std::string error;
        CPP_LLM_REQUIRE(suite, dequantize_q4_k(
            weights.data(), weights.size(), 256, &decoded, &error));
        CPP_LLM_REQUIRE(suite, decoded.size() == 256);
        CPP_LLM_CHECK(suite, std::all_of(decoded.begin(), decoded.end(), [](float value) {
            return near(value, 1.0f, 1e-5f);
        }));
        const std::vector<float> input(256, 2.0f);
        std::vector<float> reference;
        std::vector<float> fused;
        CPP_LLM_REQUIRE(suite, Q4KLinearReference::forward(
            weights.data(), weights.size(), 1, 256, input, &reference, &error));
        CPP_LLM_REQUIRE(suite, Q4KLinearFused::forward(
            weights.data(), weights.size(), 1, 256, input, &fused, &error));
        CPP_LLM_CHECK(suite, near(reference[0], 512.0f));
        CPP_LLM_CHECK(suite, near(fused[0], reference[0]));
    });

    suite.run("Q4_K scalar policy executes and rejects invalid selection", [&] {
        const auto weights = make_unit_q4k_block();
        const std::vector<float> input(256, 2.0f);
        std::vector<float> output(1, 0.0f);
        const cpp_llm_engine::core::LinearRequest request{
            weights.data(), weights.size(), input.data(), input.size(),
            output.data(), output.size(), 1, 256,
            cpp_llm_engine::core::WeightFormat::Q4K};
        CPP_LLM_REQUIRE(suite, compute_quantized_linear(request, {}).is_ok());
        CPP_LLM_CHECK(suite, near(output[0], 512.0f));
        const QuantizedLinearPolicy missing_avx2{
            QuantizedLinearBackendKind::Avx2Fused,
            QuantizedLinearBackendKind::ScalarFused};
        CPP_LLM_CHECK(suite, !validate_quantized_linear_policy(missing_avx2).is_ok());
        const QuantizedLinearPolicy invalid{
            static_cast<QuantizedLinearBackendKind>(255),
            QuantizedLinearBackendKind::ScalarFused};
        const auto invalid_status = compute_quantized_linear(request, invalid);
        CPP_LLM_CHECK(suite, !invalid_status.is_ok());
        CPP_LLM_CHECK(suite, invalid_status.message() == "Q4_K backend kind is invalid");
    });

    suite.run("Q4_K multi-block Linear and malformed payload are distinguished", [&] {
        const auto block = make_unit_q4k_block();
        std::vector<std::uint8_t> weights(block.size() * 2, 0);
        std::copy(block.begin(), block.end(), weights.begin());
        std::copy(block.begin(), block.end(), weights.begin() + block.size());
        weights[block.size() + 1] = 0x40;
        const std::vector<float> input(512, 1.0f);
        std::vector<float> reference;
        std::vector<float> fused;
        std::string error;
        CPP_LLM_REQUIRE(suite, Q4KLinearReference::forward(
            weights.data(), weights.size(), 1, 512, input, &reference, &error));
        CPP_LLM_REQUIRE(suite, Q4KLinearFused::forward(
            weights.data(), weights.size(), 1, 512, input, &fused, &error));
        CPP_LLM_CHECK(suite, near(reference[0], 768.0f));
        CPP_LLM_CHECK(suite, near(fused[0], reference[0]));
        std::vector<float> rejected;
        const std::vector<float> overflow_input(256, 1.0f);
        CPP_LLM_CHECK(suite, !Q4KLinearFused::forward(
            block.data(), block.size(), std::numeric_limits<std::size_t>::max(),
            256, overflow_input, &rejected, &error));
        CPP_LLM_CHECK(suite, error == "Q4_K Linear payload is too large");
    });

#ifdef CHAPTER05_HAS_AVX2
    suite.run("Q4_K AVX2 matches scalar reference when supported", [&] {
        if (!q4k_avx2_supported()) return;
        auto weights = make_unit_q4k_block();
        weights[1] = 0x38;
        weights[3] = 0x34;
        for (std::size_t i = 0; i < 12; ++i) {
            weights[4 + i] = static_cast<std::uint8_t>(3 + i * 7);
        }
        for (std::size_t i = 0; i < 128; ++i) {
            weights[16 + i] = static_cast<std::uint8_t>((i * 29 + 11) & 0xff);
        }
        std::vector<float> input(256, 0.0f);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 7.0f;
        }
        std::vector<float> reference;
        std::vector<float> avx2;
        std::string error;
        CPP_LLM_REQUIRE(suite, Q4KLinearReference::forward(
            weights.data(), weights.size(), 1, 256, input, &reference, &error));
        CPP_LLM_REQUIRE(suite, Q4KLinearAvx2::forward(
            weights.data(), weights.size(), 1, 256, input, &avx2, &error));
        CPP_LLM_CHECK(suite, near(avx2[0], reference[0], 1e-2f));
    });
#endif

    suite.run("Q6_K block decoding rejects malformed payload", [&] {
        const auto block = make_q6k_block();
        std::vector<float> output;
        std::string error;
        CPP_LLM_REQUIRE(suite, dequantize_q6_k(
            block.data(), block.size(), 256, &output, &error));
        CPP_LLM_CHECK(suite, near(output[0], -15.0f, 1e-5f));
        CPP_LLM_CHECK(suite, near(output[32], 3.0f, 1e-5f));
        CPP_LLM_CHECK(suite, near(output[64], 18.0f, 1e-5f));
        CPP_LLM_CHECK(suite, near(output[96], -28.0f, 1e-5f));
        CPP_LLM_CHECK(suite, near(output[127], -32.0f, 1e-5f));
        CPP_LLM_CHECK(suite, !dequantize_q6_k(
            block.data(), block.size() - 1, 256, &output, &error));
        CPP_LLM_CHECK(suite, error == "Q6_K byte count does not match block count");
    });

    suite.run("Q6_K fused Linear and scalar policy match reference", [&] {
        const auto weights = make_q6k_block();
        const std::vector<float> input(256, 1.0f);
        std::vector<float> reference;
        std::vector<float> fused;
        std::string error;
        CPP_LLM_REQUIRE(suite, Q6KLinearReference::forward(
            weights.data(), weights.size(), 1, 256, input, &reference, &error));
        CPP_LLM_REQUIRE(suite, Q6KLinearFused::forward(
            weights.data(), weights.size(), 1, 256, input, &fused, &error));
        CPP_LLM_CHECK(suite, near(reference[0], -8086.0f));
        CPP_LLM_CHECK(suite, near(fused[0], reference[0]));
        std::vector<float> output(1, 0.0f);
        const cpp_llm_engine::core::LinearRequest request{
            weights.data(), weights.size(), input.data(), input.size(),
            output.data(), output.size(), 1, 256,
            cpp_llm_engine::core::WeightFormat::Q6K};
        CPP_LLM_REQUIRE(suite, compute_quantized_linear(request, {}).is_ok());
        CPP_LLM_CHECK(suite, near(output[0], reference[0]));
    });

#ifdef CHAPTER05_HAS_AVX2
    suite.run("Q6_K AVX2 matches scalar reference when supported", [&] {
        if (!q6k_avx2_supported()) return;
        const auto weights = make_q6k_block();
        const std::vector<float> input(256, 1.0f);
        std::vector<float> reference;
        std::vector<float> avx2;
        std::string error;
        CPP_LLM_REQUIRE(suite, Q6KLinearReference::forward(
            weights.data(), weights.size(), 1, 256, input, &reference, &error));
        CPP_LLM_REQUIRE(suite, Q6KLinearAvx2::forward(
            weights.data(), weights.size(), 1, 256, input, &avx2, &error));
        CPP_LLM_CHECK(suite, near(avx2[0], reference[0]));
    });
#endif

    return suite.finish();
}
