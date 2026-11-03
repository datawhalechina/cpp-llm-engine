// Engine/Chapter 08：真实 TinyLlama CPU Decoder 子阶段 profiling。
// 职责：显式开启 Context 计时，汇总量化投影、attention、FFN 和 LM Head 耗时。
// 运行：real_tinyllama_decoder_profile <model.gguf> [iterations] [backend]。
#include "cpp_llm_engine/real_tokenizer_backend.h"
#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"
#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void add_profile(
    chapter01::RealLlamaDecoderProfile* total,
    const chapter01::RealLlamaDecoderProfile& value) {
    total->forward_tokens += value.forward_tokens;
    total->total_ms += value.total_ms;
    total->embedding_ms += value.embedding_ms;
    total->normalization_ms += value.normalization_ms;
    total->qkv_projection_ms += value.qkv_projection_ms;
    total->rope_ms += value.rope_ms;
    total->attention_ms += value.attention_ms;
    total->attention_output_ms += value.attention_output_ms;
    total->ffn_ms += value.ffn_ms;
    total->ffn_gate_ms += value.ffn_gate_ms;
    total->ffn_up_ms += value.ffn_up_ms;
    total->ffn_activation_ms += value.ffn_activation_ms;
    total->ffn_down_ms += value.ffn_down_ms;
    total->ffn_residual_ms += value.ffn_residual_ms;
    total->lm_head_ms += value.lm_head_ms;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: real_tinyllama_decoder_profile <model.gguf> "
                     "[iterations] [scalar|q4-avx2|q6-avx2|"
                     "q4-attention-avx2|optimized-avx2]\n";
        return 2;
    }
    const std::size_t iterations = argc >= 3
        ? static_cast<std::size_t>(std::strtoul(argv[2], nullptr, 10)) : 1;
    if (iterations == 0) return 2;
    const std::string backend = argc == 4 ? argv[3] : "scalar";
    if (backend != "scalar" && backend != "q4-avx2" &&
        backend != "q6-avx2" && backend != "avx2" &&
        backend != "q4-attention-avx2" && backend != "optimized-avx2") return 2;

    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto model = chapter01::RealLlamaCpuModel::load(argv[1], 0, &error);
    if (tokenizer == nullptr || model == nullptr) {
        std::cerr << "decoder profile load failed: " << error << '\n';
        return 1;
    }

    const auto prompt = tokenizer->encode("Hello TinyLlama");
    const chapter06_generation::TokenSequence expected{29892, 306, 29915, 29885};
    chapter06_generation::GreedySampler sampler;
    chapter01::RealLlamaDecoderProfile total;
    bool correctness = true;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        auto context = model->create_context();
        const chapter05_quantization::QuantizedLinearPolicy attention_policy{
            backend == "q4-attention-avx2" || backend == "optimized-avx2"
                ? chapter05_quantization::QuantizedLinearBackendKind::Avx2Fused
                : chapter05_quantization::QuantizedLinearBackendKind::ScalarFused,
            chapter05_quantization::QuantizedLinearBackendKind::ScalarFused};
        const chapter05_quantization::QuantizedLinearPolicy ffn_policy{
            backend == "q4-avx2" || backend == "optimized-avx2"
                ? chapter05_quantization::QuantizedLinearBackendKind::Avx2Fused
                : chapter05_quantization::QuantizedLinearBackendKind::ScalarFused,
            backend == "q6-avx2" || backend == "avx2" ||
                backend == "optimized-avx2"
                ? chapter05_quantization::QuantizedLinearBackendKind::Avx2Fused
                : chapter05_quantization::QuantizedLinearBackendKind::ScalarFused};
        if (!context->set_attention_linear_policy(attention_policy, &error)) {
            std::cerr << "decoder profile Attention backend policy failed: "
                      << error << '\n';
            return 1;
        }
        if (!context->set_ffn_linear_policy(ffn_policy, &error)) {
            std::cerr << "decoder profile FFN backend policy failed: " << error << '\n';
            return 1;
        }
        context->set_profiling_enabled(true);
        chapter06_generation::Logits logits;
        chapter06_generation::TokenSequence generated;
        if (!context->prefill(prompt, &logits, &error)) {
            std::cerr << "decoder profile prefill failed: " << error << '\n';
            return 1;
        }
        for (std::size_t step = 0; step < expected.size(); ++step) {
            const auto token = sampler.sample(logits);
            generated.push_back(token);
            if (step + 1 < expected.size() && !context->decode(token, &logits, &error)) {
                std::cerr << "decoder profile decode failed: " << error << '\n';
                return 1;
            }
        }
        correctness = correctness && generated == expected;
        add_profile(&total, context->profile());
    }

    const double divisor = static_cast<double>(iterations);
    const double measured = total.embedding_ms + total.normalization_ms +
        total.qkv_projection_ms + total.rope_ms + total.attention_ms +
        total.attention_output_ms + total.ffn_ms + total.lm_head_ms;
    const double ffn_measured = total.ffn_gate_ms + total.ffn_up_ms +
        total.ffn_activation_ms + total.ffn_down_ms + total.ffn_residual_ms;
    const bool profile_complete =
        total.forward_tokens == iterations * 9 && total.total_ms > 0.0 &&
        measured <= total.total_ms * 1.01;
    const bool ffn_profile_complete =
        total.ffn_ms > 0.0 && ffn_measured > 0.0 &&
        ffn_measured <= total.ffn_ms * 1.01;
    std::cout << std::fixed << std::setprecision(4)
              << "decoder_profile=tinyllama_cpu_subphases\n"
              << "correctness=" << (correctness ? "PASS" : "FAIL") << '\n'
              << "profile_complete=" << (profile_complete ? "PASS" : "FAIL") << '\n'
              << "ffn_profile_complete="
              << (ffn_profile_complete ? "PASS" : "FAIL") << '\n'
              << "attention_q4_backend="
              << (backend == "q4-attention-avx2" || backend == "optimized-avx2"
                      ? "avx2" : "scalar") << '\n'
              << "ffn_q4_backend="
              << (backend == "q4-avx2" || backend == "optimized-avx2"
                      ? "avx2" : "scalar") << '\n'
              << "ffn_down_backend="
              << (backend == "q6-avx2" || backend == "avx2" ||
                      backend == "optimized-avx2" ? "avx2" : "scalar")
              << '\n'
              << "lm_head_backend=scalar\n"
              << "iterations=" << iterations << '\n'
              << "forward_tokens_per_iteration="
              << total.forward_tokens / iterations << '\n'
              << "total_ms=" << total.total_ms / divisor << '\n'
              << "embedding_ms=" << total.embedding_ms / divisor << '\n'
              << "normalization_ms=" << total.normalization_ms / divisor << '\n'
              << "qkv_projection_ms=" << total.qkv_projection_ms / divisor << '\n'
              << "rope_ms=" << total.rope_ms / divisor << '\n'
              << "attention_ms=" << total.attention_ms / divisor << '\n'
              << "attention_output_ms=" << total.attention_output_ms / divisor << '\n'
              << "ffn_ms=" << total.ffn_ms / divisor << '\n'
              << "ffn_gate_ms=" << total.ffn_gate_ms / divisor << '\n'
              << "ffn_up_ms=" << total.ffn_up_ms / divisor << '\n'
              << "ffn_activation_ms=" << total.ffn_activation_ms / divisor << '\n'
              << "ffn_down_ms=" << total.ffn_down_ms / divisor << '\n'
              << "ffn_residual_ms=" << total.ffn_residual_ms / divisor << '\n'
              << "ffn_unattributed_ms="
              << std::max(0.0, (total.ffn_ms - ffn_measured) / divisor) << '\n'
              << "lm_head_ms=" << total.lm_head_ms / divisor << '\n'
              << "unattributed_ms="
              << std::max(0.0, (total.total_ms - measured) / divisor) << '\n';
    return correctness && profile_complete && ffn_profile_complete ? 0 : 1;
}
