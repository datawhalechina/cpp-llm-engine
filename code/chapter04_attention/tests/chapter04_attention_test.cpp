// Chapter 04：Attention/Decoder 综合回归测试
// 验证因果遮罩、decode offset、KV Cache、LayerNorm、FFN 和 DecoderLayer。
// 同时检查 backend 生命周期约束和非法 Attention 参数。
// 验证：bash code/chapter04_attention/scripts/run.sh。
#include "chapter04_attention/decoder_layer.h"
#include "chapter04_attention/attention_backend.h"
#include "chapter04_attention/kv_cache.h"
#include "cpp_llm_engine/testing/test.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace {

chapter04_attention::Tensor make_input() {
    chapter04_attention::Tensor input(2, 4);
    input.at(0, 0) = 1.0f;
    input.at(0, 1) = 2.0f;
    input.at(0, 2) = 3.0f;
    input.at(0, 3) = 4.0f;
    input.at(1, 0) = 5.0f;
    input.at(1, 1) = 6.0f;
    input.at(1, 2) = 7.0f;
    input.at(1, 3) = 8.0f;
    return input;
}

}  // namespace

static_assert(
    !std::is_constructible_v<
        chapter04_attention::MultiHeadAttention,
        std::size_t,
        std::size_t,
        chapter04_attention::CpuReferenceAttentionBackend&&>,
    "MultiHeadAttention must reject temporary backends");

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter04_attention");
    suite.run("attention cache and decoder components preserve contracts", [&] {
    const float query[] = {1.0f, 1.0f};
    const float key[] = {1.0f, 1.0f};
    const float value[] = {2.0f, 4.0f};
    float attention_output[] = {0.0f, 0.0f};
    const cpp_llm_engine::core::AttentionRequest causal_request{
        query, key, value, attention_output, 2, 1, 1, 1, 0, true, 1.0f};
    CPP_LLM_REQUIRE(
        suite,
        chapter04_attention::cpu_reference_attention_backend()
            .compute(causal_request)
            .is_ok());
    CPP_LLM_CHECK(suite, std::abs(attention_output[0] - 2.0f) < 1e-5f);

    const float two_keys[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float two_values[] = {2.0f, 4.0f, 100.0f, 200.0f};
    float first_decode_output[] = {0.0f, 0.0f};
    const cpp_llm_engine::core::AttentionRequest future_mask_request{
        query, two_keys, two_values, first_decode_output, 1, 2, 1, 2, 0, true, 1.0f};
    CPP_LLM_REQUIRE(
        suite,
        chapter04_attention::cpu_reference_attention_backend()
            .compute(future_mask_request)
            .is_ok());
    CPP_LLM_CHECK(suite, std::abs(first_decode_output[0] - 2.0f) < 1e-5f);

    float last_position_output[] = {-1.0F, -1.0F};
    const cpp_llm_engine::core::AttentionRequest last_position_request{
        query, key, value, last_position_output, 1, 1, 1, 2,
        std::numeric_limits<std::size_t>::max(), true, 1.0F};
    CPP_LLM_CHECK(
        suite,
        chapter04_attention::cpu_reference_attention_backend()
            .compute(last_position_request)
            .is_ok());
    CPP_LLM_CHECK(suite, std::abs(last_position_output[0] - 2.0F) < 1e-5F);

    float invalid_scale_output[] = {-3.0F, -3.0F};
    auto invalid_scale_request = last_position_request;
    invalid_scale_request.output = invalid_scale_output;
    invalid_scale_request.scale = -1.0F;
    CPP_LLM_CHECK(
        suite,
        !chapter04_attention::cpu_reference_attention_backend()
             .compute(invalid_scale_request)
             .is_ok());
    CPP_LLM_CHECK(suite, invalid_scale_output[0] == -3.0F);
    invalid_scale_request.scale = std::numeric_limits<float>::quiet_NaN();
    CPP_LLM_CHECK(
        suite,
        !chapter04_attention::cpu_reference_attention_backend()
             .compute(invalid_scale_request)
             .is_ok());

    chapter04_attention::KvCache cache(2, 1, 2);
    CPP_LLM_CHECK(suite, cache.ready());
    CPP_LLM_CHECK(suite, cache.width() == 2);
    CPP_LLM_CHECK(suite, cache.append(two_keys, two_values, 2));
    CPP_LLM_CHECK(suite, cache.size() == 2);
    CPP_LLM_CHECK(suite, cache.key_data()[0] == 1.0f);
    CPP_LLM_CHECK(suite, cache.value_data()[2] == 100.0f);
    CPP_LLM_CHECK(suite, !cache.append(two_keys, two_values, 1));

    float cached_output[] = {0.0f, 0.0f};
    const cpp_llm_engine::core::AttentionRequest cached_decode_request{
        query, cache.key_data(), cache.value_data(), cached_output,
        1, cache.size(), 1, 2, 1, true, 1.0f};
    CPP_LLM_REQUIRE(
        suite,
        chapter04_attention::cpu_reference_attention_backend()
            .compute(cached_decode_request)
            .is_ok());
    CPP_LLM_CHECK(suite, std::abs(cached_output[0] - 51.0f) < 1e-5f);
    cache.clear();
    CPP_LLM_CHECK(suite, cache.size() == 0);
    CPP_LLM_CHECK(suite, cache.append(two_keys, two_values, 1));

    const chapter04_attention::Tensor input = make_input();

    chapter04_attention::LayerNorm norm(4);
    const chapter04_attention::Tensor normed = norm.forward(input);
    CPP_LLM_CHECK(suite, norm.ready());
    CPP_LLM_CHECK(suite, normed.ready());
    CPP_LLM_CHECK(suite, normed.rows == 2);
    CPP_LLM_CHECK(suite, normed.cols == 4);

    chapter04_attention::MultiHeadAttention attention(4, 2);
    const chapter04_attention::Tensor attended = attention.forward(input);
    CPP_LLM_CHECK(suite, attention.ready());
    CPP_LLM_CHECK(suite, attended.ready());
    CPP_LLM_CHECK(suite, attended.rows == 2);
    CPP_LLM_CHECK(suite, attended.cols == 4);
    CPP_LLM_CHECK(suite, attention.backend_kind() == cpp_llm_engine::core::BackendKind::CPU);

    chapter04_attention::FeedForward feed_forward(4, 8);
    const chapter04_attention::Tensor ffn_output = feed_forward.forward(input);
    CPP_LLM_CHECK(suite, feed_forward.ready());
    CPP_LLM_CHECK(suite, ffn_output.ready());
    CPP_LLM_CHECK(suite, ffn_output.rows == 2);
    CPP_LLM_CHECK(suite, ffn_output.cols == 4);

    chapter04_attention::DecoderLayer layer(4, 2, 8, 0);
    const chapter04_attention::Tensor output = layer.forward(input);
    CPP_LLM_CHECK(suite, layer.ready());
    CPP_LLM_CHECK(suite, output.ready());
    CPP_LLM_CHECK(suite, output.rows == 2);
    CPP_LLM_CHECK(suite, output.cols == 4);
    CPP_LLM_CHECK(suite, std::abs(output.at(0, 0) - input.at(0, 0)) > 1e-4f);
    CPP_LLM_CHECK(suite, std::abs(output.at(1, 3) - input.at(1, 3)) > 1e-4f);
    const auto layer_contract = layer.contract();
    CPP_LLM_CHECK(suite, layer_contract.validate().is_ok());
    CPP_LLM_CHECK(suite, layer_contract.hidden_size == 4);
    CPP_LLM_CHECK(suite, layer_contract.ffn_hidden_size == 8);
    CPP_LLM_CHECK(suite, layer_contract.query_head_count == 2);
    CPP_LLM_CHECK(suite, layer_contract.kv_head_count == 2);
    CPP_LLM_CHECK(suite, layer_contract.normalization ==
           cpp_llm_engine::core::DecoderNormalizationKind::LayerNorm);
    CPP_LLM_CHECK(suite, layer_contract.activation ==
           cpp_llm_engine::core::DecoderActivationKind::Relu);
    CPP_LLM_CHECK(suite, layer_contract.uses_kv_cache);

    });
    suite.run("decoder request contexts isolate and commit KV state", [&] {
        const chapter04_attention::DecoderLayer layer(4, 2, 8, 0);
        chapter04_attention::DecoderLayerContext first_context(3, 2, 2);
        chapter04_attention::DecoderLayerContext second_context(3, 2, 2);
        const auto prefill_input = make_input();
        chapter04_attention::Tensor prefill_output(2, 4, -1.0F);

        auto status = layer.try_prefill(
            prefill_input.view_ref(), &first_context,
            prefill_output.mutable_view_ref());
        CPP_LLM_REQUIRE(suite, status.is_ok());
        CPP_LLM_CHECK(suite, first_context.token_count() == 2);
        CPP_LLM_CHECK(suite, second_context.token_count() == 0);

        chapter04_attention::Tensor decode_input(1, 4);
        decode_input.at(0, 0) = 9.0F;
        decode_input.at(0, 1) = 10.0F;
        decode_input.at(0, 2) = 11.0F;
        decode_input.at(0, 3) = 12.0F;
        chapter04_attention::Tensor decode_output(1, 4, -2.0F);
        status = layer.try_decode(
            decode_input.view_ref(), &first_context,
            decode_output.mutable_view_ref());
        CPP_LLM_REQUIRE(suite, status.is_ok());
        CPP_LLM_CHECK(suite, first_context.token_count() == 3);
        CPP_LLM_CHECK(suite, second_context.token_count() == 0);

        chapter04_attention::Tensor failed_output(1, 4, -3.0F);
        status = layer.try_decode(
            decode_input.view_ref(), &first_context,
            failed_output.mutable_view_ref());
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, first_context.token_count() == 3);
        CPP_LLM_CHECK(suite, failed_output.at(0, 0) == -3.0F);

        status = layer.try_decode(
            decode_input.view_ref(), &second_context,
            failed_output.mutable_view_ref());
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, second_context.token_count() == 0);
        CPP_LLM_CHECK(suite, failed_output.at(0, 0) == -3.0F);

        first_context.reset();
        CPP_LLM_CHECK(suite, first_context.token_count() == 0);
        CPP_LLM_CHECK(suite, first_context.capacity() == 3);
    });
    return suite.finish();
}
