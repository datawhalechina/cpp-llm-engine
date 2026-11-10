// Chapter 05：DeepSeek 固定 GGUF 使用的 Q2_K 与 IQ4_NL 测试。
// 职责：核对 block 解码、统一 Linear policy 和失败时输出不变。
// 验证：bash code/chapter05_quantization/scripts/run.sh。
#include "chapter05_quantization/iq4nl_dequantize.h"
#include "chapter05_quantization/q2k_dequantize.h"
#include "chapter05_quantization/quantized_linear_policy.h"
#include "cpp_llm_engine/testing/test.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

void write_half(std::vector<std::uint8_t>* bytes,
                std::size_t offset,
                std::uint16_t bits) {
    std::memcpy(bytes->data() + offset, &bits, sizeof(bits));
}

bool close(float left, float right, float tolerance = 1.0e-6F) {
    return std::abs(left - right) <=
        tolerance * std::max(1.0F, std::abs(right));
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter05_deepseek_quantization");

    suite.run("Q2_K restores packed quants, scales and minimums", [&] {
        std::vector<std::uint8_t> block(
            chapter05_quantization::kQ2KBlockBytes, 0);
        std::fill(block.begin(), block.begin() + 16, 0x21u);
        std::fill(block.begin() + 16, block.begin() + 80, 0xe4u);
        write_half(&block, 80, 0x3c00u);  // 1.0
        write_half(&block, 82, 0x3800u);  // 0.5

        std::vector<float> decoded;
        std::string error;
        CPP_LLM_REQUIRE(
            suite,
            chapter05_quantization::dequantize_q2_k(
                block.data(), block.size(),
                chapter05_quantization::kQ2KBlockElements,
                &decoded, &error));
        CPP_LLM_CHECK(suite, close(decoded[0], -1.0F));
        CPP_LLM_CHECK(suite, close(decoded[31], -1.0F));
        CPP_LLM_CHECK(suite, close(decoded[32], 0.0F));
        CPP_LLM_CHECK(suite, close(decoded[64], 1.0F));
        CPP_LLM_CHECK(suite, close(decoded[96], 2.0F));
        CPP_LLM_CHECK(suite, close(decoded[128], -1.0F));

        std::vector<std::uint8_t> weights;
        weights.insert(weights.end(), block.begin(), block.end());
        weights.insert(weights.end(), block.begin(), block.end());
        std::vector<float> input(
            chapter05_quantization::kQ2KBlockElements, 1.0F);
        float output[] = {-9.0F, -9.0F};
        const cpp_llm_engine::core::LinearRequest request{
            weights.data(), weights.size(), input.data(), input.size(),
            output, 2, 2, input.size(),
            cpp_llm_engine::core::WeightFormat::Q2K};
        CPP_LLM_REQUIRE(
            suite,
            chapter05_quantization::compute_quantized_linear(
                request, {}).is_ok());
        CPP_LLM_CHECK(suite, close(output[0], 128.0F));
        CPP_LLM_CHECK(suite, close(output[1], 128.0F));
    });

    suite.run("IQ4_NL uses the non-linear lookup table", [&] {
        std::vector<std::uint8_t> block(
            chapter05_quantization::kIQ4NLBlockBytes, 0xf0u);
        write_half(&block, 0, 0x3800u);  // 0.5

        std::vector<float> decoded;
        std::string error;
        CPP_LLM_REQUIRE(
            suite,
            chapter05_quantization::dequantize_iq4_nl(
                block.data(), block.size(),
                chapter05_quantization::kIQ4NLBlockElements,
                &decoded, &error));
        CPP_LLM_CHECK(suite, close(decoded[0], -63.5F));
        CPP_LLM_CHECK(suite, close(decoded[15], -63.5F));
        CPP_LLM_CHECK(suite, close(decoded[16], 56.5F));
        CPP_LLM_CHECK(suite, close(decoded[31], 56.5F));

        std::vector<float> input(
            chapter05_quantization::kIQ4NLBlockElements, 1.0F);
        float output = -9.0F;
        const cpp_llm_engine::core::LinearRequest request{
            block.data(), block.size(), input.data(), input.size(),
            &output, 1, 1, input.size(),
            cpp_llm_engine::core::WeightFormat::IQ4_NL};
        CPP_LLM_REQUIRE(
            suite,
            chapter05_quantization::compute_quantized_linear(
                request, {}).is_ok());
        CPP_LLM_CHECK(suite, close(output, -112.0F));
    });

    suite.run("invalid Q2_K shape preserves caller output", [&] {
        std::vector<std::uint8_t> weights(
            chapter05_quantization::kQ2KBlockBytes, 0);
        std::vector<float> input(255, 1.0F);
        float output = -7.0F;
        const cpp_llm_engine::core::LinearRequest request{
            weights.data(), weights.size(), input.data(), input.size(),
            &output, 1, 1, input.size(),
            cpp_llm_engine::core::WeightFormat::Q2K};
        CPP_LLM_CHECK(
            suite,
            !chapter05_quantization::compute_quantized_linear(
                request, {}).is_ok());
        CPP_LLM_CHECK(suite, output == -7.0F);
    });

    return suite.finish();
}
