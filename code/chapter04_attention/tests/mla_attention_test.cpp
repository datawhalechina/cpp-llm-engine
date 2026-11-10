// Chapter 04：MLA 压缩缓存与数值参考测试。
// 职责：将低秩缓存路径与显式展开的标准多头 Attention 对照。
// 失败路径：非法位置必须返回错误且保留调用方输出。
// 验证：bash code/chapter04_attention/scripts/run.sh。
#include "chapter04_attention/attention_kernel.h"
#include "chapter04_attention/mla_attention.h"
#include "cpp_llm_engine/testing/test.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

bool close(float left, float right, float tolerance = 2.0e-5F) {
    return std::abs(left - right) <=
        tolerance * std::max(1.0F, std::abs(right));
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter04_mla_attention");

    suite.run("compressed MLA cache matches explicitly expanded attention", [&] {
        constexpr std::size_t query_tokens = 2;
        constexpr std::size_t key_tokens = 3;
        constexpr std::size_t heads = 2;
        constexpr std::size_t nope_dim = 1;
        constexpr std::size_t rope_dim = 1;
        constexpr std::size_t latent_dim = 2;
        constexpr std::size_t value_dim = 2;
        constexpr std::size_t qk_dim = nope_dim + rope_dim;

        const std::vector<float> query_nope{0.5F, -0.25F, 0.75F, 0.4F};
        const std::vector<float> query_rope{0.2F, 0.1F, -0.3F, 0.5F};
        const std::vector<float> latent{
            1.0F, 0.5F,
            -0.5F, 1.5F,
            0.25F, -1.0F,
        };
        const std::vector<float> key_rope{0.4F, -0.2F, 0.6F};
        const std::vector<float> key_up{
            1.0F, -0.5F,
            0.25F, 0.75F,
        };
        const std::vector<float> value_up{
            1.0F, 0.0F,
            0.0F, 1.0F,
            0.5F, 0.5F,
            -0.25F, 1.0F,
        };
        std::vector<float> actual(query_tokens * heads * value_dim, -9.0F);
        const chapter04_attention::MlaAttentionRequest request{
            query_nope.data(), query_rope.data(), latent.data(),
            key_rope.data(), key_up.data(), value_up.data(), actual.data(),
            query_tokens, key_tokens, 1, heads, nope_dim, rope_dim,
            latent_dim, value_dim, true, 0.0F};
        CPP_LLM_REQUIRE(
            suite,
            chapter04_attention::cpu_mla_attention_reference(request).is_ok());

        std::vector<float> query(query_tokens * heads * qk_dim, 0.0F);
        std::vector<float> key(key_tokens * heads * qk_dim, 0.0F);
        std::vector<float> value(key_tokens * heads * value_dim, 0.0F);
        for (std::size_t token = 0; token < query_tokens; ++token) {
            for (std::size_t head = 0; head < heads; ++head) {
                const std::size_t q_offset = (token * heads + head) * qk_dim;
                query[q_offset] = query_nope[token * heads + head];
                query[q_offset + 1] = query_rope[token * heads + head];
            }
        }
        for (std::size_t token = 0; token < key_tokens; ++token) {
            for (std::size_t head = 0; head < heads; ++head) {
                const float* latent_row = latent.data() + token * latent_dim;
                const float* key_weight = key_up.data() + head * latent_dim;
                const std::size_t q_offset = (token * heads + head) * qk_dim;
                key[q_offset] = latent_row[0] * key_weight[0] +
                                latent_row[1] * key_weight[1];
                key[q_offset + 1] = key_rope[token];
                for (std::size_t dim = 0; dim < value_dim; ++dim) {
                    const float* value_weight = value_up.data() +
                        (head * value_dim + dim) * latent_dim;
                    value[(token * heads + head) * value_dim + dim] =
                        latent_row[0] * value_weight[0] +
                        latent_row[1] * value_weight[1];
                }
            }
        }
        std::vector<float> golden(actual.size(), 0.0F);
        const cpp_llm_engine::core::AttentionRequest golden_request{
            query.data(), key.data(), value.data(), golden.data(),
            query_tokens, key_tokens, heads, qk_dim, 1, true, 0.0F};
        CPP_LLM_REQUIRE(
            suite,
            chapter04_attention::cpu_reference_attention_kernel(
                golden_request).is_ok());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            CPP_LLM_CHECK(suite, close(actual[index], golden[index]));
        }

        const std::size_t compressed_cache =
            key_tokens * (latent_dim + rope_dim);
        const std::size_t expanded_cache =
            key_tokens * heads * (qk_dim + value_dim);
        CPP_LLM_CHECK(suite, compressed_cache < expanded_cache);
    });

    suite.run("invalid query positions preserve caller output", [&] {
        const float scalar = 1.0F;
        float output = -7.0F;
        const chapter04_attention::MlaAttentionRequest request{
            &scalar, &scalar, &scalar, &scalar, &scalar, &scalar, &output,
            1, 1, 1, 1, 1, 1, 1, 1, true, 1.0F};
        const auto status =
            chapter04_attention::cpu_mla_attention_reference(request);
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, output == -7.0F);
    });

    return suite.finish();
}
