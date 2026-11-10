// Model：验证 Qwen2MoE 单层与最终输出契约。
// 职责：以小尺寸 Q5_1 权重核对 GQA、K/V、LM Head 和失败原子性。
// 通过条件：双词元和 logits 有限，非法宽度不覆盖原结果。
#include "cpp_llm_engine/model/real_qwen2_moe_layer.h"
#include "cpp_llm_engine/model/real_qwen2_moe_output.h"
#include "cpp_llm_engine/testing/test.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

std::vector<std::byte> q5_1_ones(std::size_t block_count) {
    constexpr std::size_t kBlockBytes = 24;
    std::vector<std::byte> bytes(block_count * kBlockBytes, std::byte{0});
    for (std::size_t block = 0; block < block_count; ++block) {
        auto* raw = reinterpret_cast<std::uint8_t*>(
            bytes.data() + block * kBlockBytes);
        const std::uint16_t one = 0x3c00;
        std::memcpy(raw, &one, sizeof(one));
        for (std::size_t quant = 8; quant < kBlockBytes; ++quant) {
            raw[quant] = 0x11;
        }
    }
    return bytes;
}

chapter01::Tensor q5_1_tensor(const char* name,
                              std::vector<std::int64_t> shape) {
    std::size_t blocks = 1;
    for (std::size_t dimension = 1; dimension < shape.size(); ++dimension) {
        blocks *= static_cast<std::size_t>(shape[dimension]);
    }
    blocks *= static_cast<std::size_t>(shape[0]) / 32;
    return chapter01::Tensor::from_raw_bytes(
        name, std::move(shape), chapter01::DType::Q5_1,
        q5_1_ones(blocks));
}

chapter01::RealMoeLayerWeights make_weights() {
    chapter01::RealMoeLayerWeights weights;
    weights.attn_norm = chapter01::Tensor::from_vector<float>(
        "attn-norm", {32}, std::vector<float>(32, 1.0F));
    weights.attn_q = q5_1_tensor("q", {32, 32});
    weights.attn_k = q5_1_tensor("k", {32, 16});
    weights.attn_v = q5_1_tensor("v", {32, 16});
    weights.attn_output = q5_1_tensor("o", {32, 32});
    weights.attn_q_bias = chapter01::Tensor::from_vector<float>(
        "q-bias", {32}, std::vector<float>(32, 0.0F));
    weights.attn_k_bias = chapter01::Tensor::from_vector<float>(
        "k-bias", {16}, std::vector<float>(16, 0.0F));
    weights.attn_v_bias = chapter01::Tensor::from_vector<float>(
        "v-bias", {16}, std::vector<float>(16, 0.0F));
    weights.ffn_norm = chapter01::Tensor::from_vector<float>(
        "ffn-norm", {32}, std::vector<float>(32, 1.0F));

    std::vector<float> router(64, 0.0F);
    router[0] = 1.0F;
    router[32] = 2.0F;
    weights.router = chapter01::Tensor::from_vector<float>(
        "router", {32, 2}, router);
    weights.shared_expert_router = chapter01::Tensor::from_vector<float>(
        "shared-router", {32}, std::vector<float>(32, 0.0F));
    weights.expert_gate = q5_1_tensor("expert-gate", {32, 32, 2});
    weights.expert_up = q5_1_tensor("expert-up", {32, 32, 2});
    weights.expert_down = q5_1_tensor("expert-down", {32, 32, 2});
    weights.shared_expert_gate = q5_1_tensor("shared-gate", {32, 32});
    weights.shared_expert_up = q5_1_tensor("shared-up", {32, 32});
    weights.shared_expert_down = q5_1_tensor("shared-down", {32, 32});
    return weights;
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("real_qwen2_moe_layer");

    chapter01::ModelMetadata metadata;
    metadata.architecture = "qwen2moe";
    metadata.hidden_size = 32;
    metadata.attention_head_count = 2;
    metadata.attention_head_count_kv = 1;
    metadata.rope_freq_base = 10000.0F;
    metadata.rms_norm_epsilon = 1.0e-5F;
    metadata.expert_count = 2;
    metadata.expert_used_count = 1;
    const auto weights = make_weights();

    suite.run("two tokens produce grouped K/V payload", [&] {
        std::vector<std::vector<float>> inputs(
            2, std::vector<float>(32, 0.0F));
        inputs[0][0] = 1.0F;
        inputs[1][1] = 2.0F;
        cpp_llm_engine::model::RealQwen2MoePrefillResult result;
        std::string error;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_qwen2_moe_prefill_layer(
                       metadata, weights, inputs, &result, &error));
        CPP_LLM_REQUIRE(suite, result.tokens.size() == 2);
        CPP_LLM_CHECK(suite, result.keys.size() == 32);
        CPP_LLM_CHECK(suite, result.values.size() == 32);
        for (const auto& token : result.tokens) {
            CPP_LLM_REQUIRE(suite, token.output.size() == 32);
            for (const float value : token.output) {
                CPP_LLM_CHECK(suite, std::isfinite(value));
            }
        }
    });

    suite.run("request context matches one-shot causal execution", [&] {
        std::vector<std::vector<float>> inputs(
            2, std::vector<float>(32, 0.0F));
        inputs[0][0] = 1.0F;
        inputs[1][1] = 2.0F;

        cpp_llm_engine::model::RealQwen2MoePrefillResult one_shot;
        cpp_llm_engine::model::RealQwen2MoeLayerContext context;
        cpp_llm_engine::model::RealQwen2MoePrefillResult first;
        cpp_llm_engine::model::RealQwen2MoePrefillResult second;
        std::string error;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_qwen2_moe_prefill_layer(
                       metadata, weights, inputs, &one_shot, &error));
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_qwen2_moe_layer_with_context(
                       metadata, weights, {inputs[0]}, &context, &first,
                       &error));
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_qwen2_moe_layer_with_context(
                       metadata, weights, {inputs[1]}, &context, &second,
                       &error));

        CPP_LLM_REQUIRE(suite, one_shot.tokens.size() == 2);
        CPP_LLM_REQUIRE(suite, second.tokens.size() == 1);
        CPP_LLM_CHECK(suite, context.position == 2);
        CPP_LLM_CHECK(suite, context.keys.size() == 32);
        CPP_LLM_CHECK(suite, context.values.size() == 32);
        for (std::size_t i = 0; i < 32; ++i) {
            CPP_LLM_CHECK(
                suite, std::fabs(one_shot.tokens[1].output[i] -
                                 second.tokens[0].output[i]) <= 1.0e-5F);
        }

        const auto saved_keys = context.keys;
        const auto saved_values = context.values;
        cpp_llm_engine::model::RealQwen2MoePrefillResult failed;
        CPP_LLM_CHECK(
            suite, !cpp_llm_engine::model::run_qwen2_moe_layer_with_context(
                       metadata, weights, {std::vector<float>(31, 0.0F)},
                       &context, &failed, &error));
        CPP_LLM_CHECK(suite, context.position == 2);
        CPP_LLM_CHECK(suite, context.keys == saved_keys);
        CPP_LLM_CHECK(suite, context.values == saved_values);

        context.reset();
        CPP_LLM_CHECK(suite, context.position == 0);
        CPP_LLM_CHECK(suite, context.keys.empty());
        CPP_LLM_CHECK(suite, context.values.empty());
    });

    suite.run("request batch groups MoE while preserving separate K/V", [&] {
        std::vector<float> prefix_a(32, 0.0F);
        std::vector<float> prefix_b(32, 0.0F);
        std::vector<float> next_a(32, 0.0F);
        std::vector<float> next_b(32, 0.0F);
        prefix_a[0] = 1.0F;
        prefix_b[1] = 1.0F;
        next_a[0] = 2.0F;
        next_b[1] = 2.0F;

        cpp_llm_engine::model::RealQwen2MoeLayerContext sequential_a;
        cpp_llm_engine::model::RealQwen2MoeLayerContext sequential_b;
        cpp_llm_engine::model::RealQwen2MoeLayerContext batched_a;
        cpp_llm_engine::model::RealQwen2MoeLayerContext batched_b;
        cpp_llm_engine::model::RealQwen2MoePrefillResult ignored;
        std::string error;
        for (auto* context : {&sequential_a, &batched_a}) {
            CPP_LLM_REQUIRE(
                suite, cpp_llm_engine::model::run_qwen2_moe_layer_with_context(
                           metadata, weights, {prefix_a}, context, &ignored,
                           &error));
        }
        for (auto* context : {&sequential_b, &batched_b}) {
            CPP_LLM_REQUIRE(
                suite, cpp_llm_engine::model::run_qwen2_moe_layer_with_context(
                           metadata, weights, {prefix_b}, context, &ignored,
                           &error));
        }

        cpp_llm_engine::model::RealQwen2MoePrefillResult sequential_result_a;
        cpp_llm_engine::model::RealQwen2MoePrefillResult sequential_result_b;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_qwen2_moe_layer_with_context(
                       metadata, weights, {next_a}, &sequential_a,
                       &sequential_result_a, &error));
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_qwen2_moe_layer_with_context(
                       metadata, weights, {next_b}, &sequential_b,
                       &sequential_result_b, &error));

        std::vector<cpp_llm_engine::model::RealQwen2MoeLayerResult> batch;
        cpp_llm_engine::model::RealMoeDispatchStats stats;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_qwen2_moe_layer_batch_with_context(
                       metadata, weights, {next_a, next_b},
                       {&batched_a, &batched_b}, &batch, &stats, &error));
        CPP_LLM_REQUIRE(suite, batch.size() == 2);
        CPP_LLM_CHECK(suite, batched_a.position == 2);
        CPP_LLM_CHECK(suite, batched_b.position == 2);
        CPP_LLM_CHECK(suite, stats.assignment_count == 2);
        CPP_LLM_CHECK(suite, stats.active_expert_count >= 1);
        CPP_LLM_CHECK(suite, stats.active_expert_count <= 2);
        CPP_LLM_CHECK(suite, stats.maximum_tokens_per_expert >= 1);
        for (std::size_t column = 0; column < 32; ++column) {
            CPP_LLM_CHECK(
                suite, std::fabs(batch[0].output[column] -
                                 sequential_result_a.tokens[0].output[column]) <=
                    1.0e-5F);
            CPP_LLM_CHECK(
                suite, std::fabs(batch[1].output[column] -
                                 sequential_result_b.tokens[0].output[column]) <=
                    1.0e-5F);
        }

        const auto saved_a = batched_a;
        const auto saved_b = batched_b;
        std::vector<cpp_llm_engine::model::RealQwen2MoeLayerResult> sentinel(1);
        CPP_LLM_CHECK(
            suite, !cpp_llm_engine::model::run_qwen2_moe_layer_batch_with_context(
                       metadata, weights, {next_a, next_b},
                       {&batched_a, nullptr}, &sentinel, nullptr, &error));
        CPP_LLM_CHECK(suite, batched_a.position == saved_a.position);
        CPP_LLM_CHECK(suite, batched_a.keys == saved_a.keys);
        CPP_LLM_CHECK(suite, batched_b.position == saved_b.position);
        CPP_LLM_CHECK(suite, sentinel.size() == 1);
    });

    suite.run("shape failure preserves prior result", [&] {
        cpp_llm_engine::model::RealQwen2MoePrefillResult result;
        result.keys = {7.0F};
        std::string error;
        CPP_LLM_CHECK(
            suite, !cpp_llm_engine::model::run_qwen2_moe_prefill_layer(
                       metadata, weights,
                       {std::vector<float>(31, 0.0F)}, &result, &error));
        CPP_LLM_CHECK(suite, result.keys == std::vector<float>{7.0F});
    });

    suite.run("final norm and lm head produce vocabulary logits", [&] {
        auto output_norm = chapter01::Tensor::from_vector<float>(
            "output-norm", {32}, std::vector<float>(32, 1.0F));
        auto output_weight = q5_1_tensor("output", {32, 2});
        std::vector<float> hidden(32, 0.0F);
        hidden[0] = 1.0F;
        std::vector<float> normalized;
        std::vector<float> logits;
        std::string error;
        chapter01::ModelMetadata output_metadata = metadata;
        output_metadata.vocab_size = 2;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::finish_qwen2_moe_output(
                       output_metadata, output_norm, output_weight, hidden,
                       &normalized, &logits, &error));
        CPP_LLM_CHECK(suite, normalized.size() == 32);
        CPP_LLM_REQUIRE(suite, logits.size() == 2);
        CPP_LLM_CHECK(suite, std::isfinite(logits[0]));
        CPP_LLM_CHECK(suite, std::fabs(logits[0] - logits[1]) <= 1.0e-6F);
    });

    return suite.finish();
}
