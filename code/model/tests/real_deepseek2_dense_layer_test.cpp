// Model：DeepSeek2 首个稠密层组件测试。
#include "cpp_llm_engine/model/real_deepseek2_dense_layer.h"
#include "chapter05_quantization/iq4nl_dequantize.h"
#include "chapter05_quantization/q2k_dequantize.h"
#include "chapter05_quantization/q3k_dequantize.h"
#include "cpp_llm_engine/testing/test.h"

#include <cstddef>
#include <vector>

namespace {

chapter01::Tensor packed(const char* name,
                         chapter01::DType dtype,
                         std::size_t cols,
                         std::size_t rows) {
    std::size_t elements_per_block = 0;
    std::size_t bytes_per_block = 0;
    if (dtype == chapter01::DType::Q2_K) {
        elements_per_block = chapter05_quantization::kQ2KBlockElements;
        bytes_per_block = chapter05_quantization::kQ2KBlockBytes;
    } else if (dtype == chapter01::DType::Q3_K) {
        elements_per_block = chapter05_quantization::kQ3KBlockElements;
        bytes_per_block = chapter05_quantization::kQ3KBlockBytes;
    } else {
        elements_per_block = chapter05_quantization::kIQ4NLBlockElements;
        bytes_per_block = chapter05_quantization::kIQ4NLBlockBytes;
    }
    const std::size_t byte_count =
        rows * cols / elements_per_block * bytes_per_block;
    return chapter01::Tensor::from_raw_bytes(
        name,
        {static_cast<std::int64_t>(cols), static_cast<std::int64_t>(rows)},
        dtype, std::vector<std::byte>(byte_count, std::byte{0}));
}

chapter01::Tensor packed_experts(const char* name,
                                 chapter01::DType dtype,
                                 std::size_t cols,
                                 std::size_t rows,
                                 std::size_t experts) {
    std::size_t elements_per_block = 0;
    std::size_t bytes_per_block = 0;
    if (dtype == chapter01::DType::Q2_K) {
        elements_per_block = chapter05_quantization::kQ2KBlockElements;
        bytes_per_block = chapter05_quantization::kQ2KBlockBytes;
    } else {
        elements_per_block = chapter05_quantization::kIQ4NLBlockElements;
        bytes_per_block = chapter05_quantization::kIQ4NLBlockBytes;
    }
    const std::size_t byte_count =
        experts * rows * cols / elements_per_block * bytes_per_block;
    return chapter01::Tensor::from_raw_bytes(
        name,
        {static_cast<std::int64_t>(cols), static_cast<std::int64_t>(rows),
         static_cast<std::int64_t>(experts)},
        dtype, std::vector<std::byte>(byte_count, std::byte{0}));
}

chapter01::RealDeepSeek2DenseLayerWeights zero_weights() {
    constexpr std::size_t hidden = 256;
    constexpr std::size_t heads = 2;
    constexpr std::size_t key = 128;
    constexpr std::size_t rope = 64;
    constexpr std::size_t value = 128;
    constexpr std::size_t latent = 256;
    chapter01::RealDeepSeek2DenseLayerWeights weights;
    weights.attn_norm = chapter01::Tensor::from_vector<float>(
        "attn_norm", {hidden}, std::vector<float>(hidden, 1.0F));
    weights.attn_q = packed("attn_q", chapter01::DType::Q2_K,
                            hidden, heads * key);
    weights.attn_kv_a_mqa = packed(
        "attn_kv_a", chapter01::DType::Q2_K, hidden, latent + rope);
    weights.attn_kv_a_norm = chapter01::Tensor::from_vector<float>(
        "attn_kv_a_norm", {latent}, std::vector<float>(latent, 1.0F));
    weights.attn_kv_b = packed(
        "attn_kv_b", chapter01::DType::Q2_K,
        latent, heads * ((key - rope) + value));
    weights.attn_output = packed(
        "attn_output", chapter01::DType::Q3_K,
        heads * value, hidden);
    weights.ffn_norm = chapter01::Tensor::from_vector<float>(
        "ffn_norm", {hidden}, std::vector<float>(hidden, 1.0F));
    weights.ffn_gate = packed(
        "ffn_gate", chapter01::DType::Q2_K, hidden, hidden);
    weights.ffn_up = packed(
        "ffn_up", chapter01::DType::Q2_K, hidden, hidden);
    weights.ffn_down = packed(
        "ffn_down", chapter01::DType::IQ4_NL, hidden, hidden);
    return weights;
}

chapter01::RealDeepSeek2MoeLayerWeights zero_moe_weights() {
    const auto dense = zero_weights();
    constexpr std::size_t hidden = 256;
    constexpr std::size_t experts = 2;
    chapter01::RealDeepSeek2MoeLayerWeights weights;
    weights.attn_norm = dense.attn_norm;
    weights.attn_q = dense.attn_q;
    weights.attn_kv_a_mqa = dense.attn_kv_a_mqa;
    weights.attn_kv_a_norm = dense.attn_kv_a_norm;
    weights.attn_kv_b = dense.attn_kv_b;
    weights.attn_output = dense.attn_output;
    weights.ffn_norm = dense.ffn_norm;
    weights.router = chapter01::Tensor::from_vector<float>(
        "router", {hidden, experts},
        std::vector<float>(hidden * experts, 0.0F));
    weights.expert_gate = packed_experts(
        "expert_gate", chapter01::DType::Q2_K,
        hidden, hidden, experts);
    weights.expert_up = packed_experts(
        "expert_up", chapter01::DType::Q2_K,
        hidden, hidden, experts);
    weights.expert_down = packed_experts(
        "expert_down", chapter01::DType::IQ4_NL,
        hidden, hidden, experts);
    weights.shared_expert_gate = packed(
        "shared_gate", chapter01::DType::Q2_K, hidden, hidden);
    weights.shared_expert_up = packed(
        "shared_up", chapter01::DType::Q2_K, hidden, hidden);
    weights.shared_expert_down = packed(
        "shared_down", chapter01::DType::IQ4_NL, hidden, hidden);
    return weights;
}

chapter01::ModelMetadata metadata() {
    chapter01::ModelMetadata value;
    value.architecture = "deepseek2";
    value.hidden_size = 256;
    value.attention_head_count = 2;
    value.attention_key_length_mla = 128;
    value.attention_value_length_mla = 128;
    value.attention_kv_lora_rank = 256;
    value.rope_dimension_count = 64;
    value.rope_freq_base = 10000.0F;
    value.rms_norm_epsilon = 1.0e-6F;
    value.expert_count = 2;
    value.expert_used_count = 1;
    value.expert_weights_normalize = false;
    return value;
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("real_deepseek2_dense_layer");
    suite.run("zero projections preserve the residual input", [&] {
        chapter01::ModelMetadata metadata;
        metadata.architecture = "deepseek2";
        metadata.hidden_size = 256;
        metadata.attention_head_count = 2;
        metadata.attention_key_length_mla = 128;
        metadata.attention_value_length_mla = 128;
        metadata.attention_kv_lora_rank = 256;
        metadata.rope_dimension_count = 64;
        metadata.rms_norm_epsilon = 1.0e-6F;
        const auto weights = zero_weights();
        std::vector<float> input(256, 0.0F);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(i % 7) * 0.1F - 0.3F;
        }
        cpp_llm_engine::model::RealDeepSeek2DenseLayerResult result;
        std::string error;
        CPP_LLM_REQUIRE(
            suite,
            cpp_llm_engine::model::run_deepseek2_dense_first_token_layer(
                metadata, weights, input, &result, &error));
        CPP_LLM_CHECK(suite, error.empty());
        CPP_LLM_CHECK(suite, result.query.size() == 256);
        CPP_LLM_CHECK(suite, result.compressed_kv_with_rope.size() == 320);
        CPP_LLM_CHECK(suite, result.normalized_latent_kv.size() == 256);
        CPP_LLM_CHECK(suite, result.attention_value.size() == 256);
        CPP_LLM_CHECK(suite, result.output == input);

        const auto preserved = result.output;
        metadata.architecture = "llama";
        CPP_LLM_CHECK(
            suite,
            !cpp_llm_engine::model::run_deepseek2_dense_first_token_layer(
                metadata, weights, input, &result, &error));
        CPP_LLM_CHECK(suite, result.output == preserved);
    });
    suite.run("Context appends compressed history atomically", [&] {
        chapter01::ModelMetadata metadata;
        metadata.architecture = "deepseek2";
        metadata.hidden_size = 256;
        metadata.attention_head_count = 2;
        metadata.attention_key_length_mla = 128;
        metadata.attention_value_length_mla = 128;
        metadata.attention_kv_lora_rank = 256;
        metadata.rope_dimension_count = 64;
        metadata.rope_freq_base = 10000.0F;
        metadata.rms_norm_epsilon = 1.0e-6F;
        const auto weights = zero_weights();
        const std::vector<float> input(256, 0.25F);
        cpp_llm_engine::model::RealDeepSeek2DenseLayerContext context;
        cpp_llm_engine::model::RealDeepSeek2DenseLayerResult result;
        std::string error;
        CPP_LLM_REQUIRE(
            suite,
            cpp_llm_engine::model::run_deepseek2_dense_token_layer(
                metadata, weights, input, &context, &result, &error));
        CPP_LLM_CHECK(suite, context.position == 1);
        CPP_LLM_CHECK(suite, context.latent_kv.size() == 256);
        CPP_LLM_CHECK(suite, context.key_rope.size() == 64);
        CPP_LLM_REQUIRE(
            suite,
            cpp_llm_engine::model::run_deepseek2_dense_token_layer(
                metadata, weights, input, &context, &result, &error));
        CPP_LLM_CHECK(suite, context.position == 2);
        CPP_LLM_CHECK(suite, context.latent_kv.size() == 512);
        CPP_LLM_CHECK(suite, context.key_rope.size() == 128);

        context.latent_kv.pop_back();
        const auto position = context.position;
        const auto latent = context.latent_kv;
        const auto key = context.key_rope;
        const auto output = result.output;
        CPP_LLM_CHECK(
            suite,
            !cpp_llm_engine::model::run_deepseek2_dense_token_layer(
                metadata, weights, input, &context, &result, &error));
        CPP_LLM_CHECK(suite, context.position == position);
        CPP_LLM_CHECK(suite, context.latent_kv == latent);
        CPP_LLM_CHECK(suite, context.key_rope == key);
        CPP_LLM_CHECK(suite, result.output == output);

        context.reset();
        CPP_LLM_CHECK(suite, context.position == 0);
        CPP_LLM_CHECK(suite, context.latent_kv.empty());
        CPP_LLM_CHECK(suite, context.key_rope.empty());
    });
    suite.run("MoE layer shares MLA state and commits route atomically", [&] {
        auto model_metadata = metadata();
        const auto weights = zero_moe_weights();
        const std::vector<float> input(256, 0.25F);
        cpp_llm_engine::model::RealDeepSeek2DenseLayerContext context;
        cpp_llm_engine::model::RealDeepSeek2DenseLayerResult result;
        cpp_llm_engine::model::RealMoeRoute route;
        cpp_llm_engine::model::RealMoeTrace trace;
        std::string error;
        CPP_LLM_REQUIRE(
            suite, cpp_llm_engine::model::run_deepseek2_moe_token_layer(
                       model_metadata, weights, input, &context, &result,
                       &route, &error, {}, &trace));
        CPP_LLM_CHECK(suite, result.output == input);
        CPP_LLM_CHECK(suite, context.position == 1);
        CPP_LLM_CHECK(suite, context.latent_kv.size() == 256);
        CPP_LLM_CHECK(suite, context.key_rope.size() == 64);
        CPP_LLM_CHECK(
            suite, route.expert_indices == std::vector<std::size_t>{0});
        CPP_LLM_CHECK(suite, route.weights.size() == 1);
        CPP_LLM_CHECK(suite, route.shared_expert_weight == 1.0F);
        CPP_LLM_CHECK(suite, trace.routed_output.size() == 256);
        CPP_LLM_CHECK(suite, trace.weighted_shared_output.size() == 256);

        const auto preserved_context = context;
        const auto preserved_output = result.output;
        const auto preserved_route = route.expert_indices;
        model_metadata.architecture = "llama";
        CPP_LLM_CHECK(
            suite, !cpp_llm_engine::model::run_deepseek2_moe_token_layer(
                       model_metadata, weights, input, &context, &result,
                       &route, &error, {}, &trace));
        CPP_LLM_CHECK(suite, context.position == preserved_context.position);
        CPP_LLM_CHECK(suite, context.latent_kv == preserved_context.latent_kv);
        CPP_LLM_CHECK(suite, context.key_rope == preserved_context.key_rope);
        CPP_LLM_CHECK(suite, result.output == preserved_output);
        CPP_LLM_CHECK(suite, route.expert_indices == preserved_route);
    });
    return suite.finish();
}
