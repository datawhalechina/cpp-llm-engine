// Chapter 06：CPU 分页 K/V 数据面的命名组件测试。
// 职责：验证分层提交、跨块 Attention 等价以及释放后的安全复用。
// 验证入口：带 component、cache 标签的 Chapter06 CTest。
#include "chapter06_generation/paged_kv_cache.h"

#include "cpp_llm_engine/testing/test.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using chapter06_generation::CpuPagedKvCache;
using cpp_llm_engine::testing::TestSuite;

std::vector<float> continuous_attention(
    const std::vector<float>& query,
    std::size_t query_heads,
    const std::vector<float>& keys,
    const std::vector<float>& values,
    std::size_t token_count,
    std::size_t kv_heads,
    std::size_t head_dim) {
    std::vector<float> output(query_heads * head_dim, 0.0f);
    std::vector<float> scores(token_count, 0.0f);
    const std::size_t queries_per_kv_head = query_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::size_t kv_head = query_head / queries_per_kv_head;
        const float* query_row = query.data() + query_head * head_dim;
        for (std::size_t token = 0; token < token_count; ++token) {
            const float* key = keys.data() + (token * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                score += query_row[dimension] * key[dimension];
            }
            scores[token] = score * scale;
        }
        const float maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        float* output_row = output.data() + query_head * head_dim;
        for (std::size_t token = 0; token < token_count; ++token) {
            const float* value = values.data() +
                (token * kv_heads + kv_head) * head_dim;
            const float probability = scores[token] / denominator;
            for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                output_row[dimension] += probability * value[dimension];
            }
        }
    }
    return output;
}

bool near(float left, float right) {
    return std::abs(left - right) < 1e-5f;
}

void write_all_layers(
    TestSuite& suite,
    CpuPagedKvCache& cache,
    std::size_t request_id,
    std::size_t token,
    const float* key,
    const float* value) {
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        CPP_LLM_CHECK(suite, cache.write_token(
            request_id,
            layer,
            token,
            key,
            value,
            cache.kv_heads() * cache.head_dim()));
    }
}

void write_three_tokens(
    TestSuite& suite,
    CpuPagedKvCache& cache,
    std::size_t request_id,
    const std::vector<float>& keys,
    const std::vector<float>& values) {
    CPP_LLM_CHECK(suite, cache.reserve(request_id, 3));
    for (std::size_t token = 0; token < 3; ++token) {
        write_all_layers(
            suite,
            cache,
            request_id,
            token,
            keys.data() + token * 2,
            values.data() + token * 2);
    }
    CPP_LLM_CHECK(suite, cache.commit_tokens(request_id, 3));
}

}  // namespace

int main() {
    TestSuite suite("chapter06_paged_kv_cache");

    suite.run("commit requires every layer and makes tokens immutable", [&] {
        CpuPagedKvCache cache(2, 4, 2, 1, 2);
        CPP_LLM_CHECK(suite, cache.payload_bytes() == 2 * 4 * 2 * 1 * 2 * 2 * sizeof(float));
        constexpr std::size_t request_id = 7;
        const float key[] = {1.0f, 0.0f};
        const float value[] = {10.0f, 1.0f};
        std::string error;

        CPP_LLM_CHECK(suite, cache.reserve(request_id, 1));
        CPP_LLM_CHECK(
            suite, cache.write_token(request_id, 0, 0, key, value, 2, &error));
        CPP_LLM_CHECK(suite, !cache.commit_tokens(request_id, 1, &error));
        CPP_LLM_CHECK(suite, error.find("every layer") != std::string::npos);
        CPP_LLM_CHECK(suite, cache.write_token(request_id, 1, 0, key, value, 2));
        CPP_LLM_CHECK(suite, cache.commit_tokens(request_id, 1));
        CPP_LLM_CHECK(
            suite, !cache.write_token(request_id, 0, 0, key, value, 2, &error));
        CPP_LLM_CHECK(suite, error.find("immutable") != std::string::npos);
    });

    suite.run("cross-block paged attention matches continuous reference", [&] {
        CpuPagedKvCache cache(2, 4, 2, 1, 2);
        constexpr std::size_t request_id = 7;
        const std::vector<float> keys{
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
        const std::vector<float> values{
            10.0f, 1.0f, 20.0f, 2.0f, 30.0f, 3.0f};
        write_three_tokens(suite, cache, request_id, keys, values);

        CPP_LLM_CHECK(suite, cache.block_pool().block_count(request_id) == 2);
        CPP_LLM_CHECK(
            suite,
            cache.block_pool().locate(request_id, 0).block !=
                cache.block_pool().locate(request_id, 2).block);
        CPP_LLM_CHECK(suite, cache.key_data(request_id, 0, 2, 0)[1] == 1.0f);
        CPP_LLM_CHECK(suite, cache.value_data(request_id, 1, 1, 0)[0] == 20.0f);

        const std::vector<float> query{1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> paged_output;
        std::string error;
        CPP_LLM_CHECK(
            suite, cache.attention(request_id, 0, query, 2, &paged_output, &error));
        const auto reference = continuous_attention(query, 2, keys, values, 3, 1, 2);
        CPP_LLM_REQUIRE(suite, paged_output.size() == reference.size());
        for (std::size_t index = 0; index < reference.size(); ++index) {
            CPP_LLM_CHECK(suite, near(paged_output[index], reference[index]));
        }
    });

    suite.run("released blocks are reusable without stale payload", [&] {
        CpuPagedKvCache cache(2, 4, 2, 1, 2);
        constexpr std::size_t request_a = 7;
        const std::vector<float> keys{
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
        const std::vector<float> values{
            10.0f, 1.0f, 20.0f, 2.0f, 30.0f, 3.0f};
        write_three_tokens(suite, cache, request_a, keys, values);
        const auto released_blocks = cache.block_pool().physical_blocks(request_a);
        CPP_LLM_CHECK(suite, cache.release(request_a));
        CPP_LLM_CHECK(suite, !cache.block_pool().contains(request_a));
        bool rejected = false;
        try {
            static_cast<void>(cache.key_data(request_a, 0, 0, 0));
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);

        constexpr std::size_t request_b = 8;
        CPP_LLM_CHECK(suite, cache.reserve(request_b, 2));
        const auto reused_blocks = cache.block_pool().physical_blocks(request_b);
        CPP_LLM_REQUIRE(suite, !released_blocks.empty());
        CPP_LLM_REQUIRE(suite, !reused_blocks.empty());
        CPP_LLM_CHECK(suite, released_blocks.front() == reused_blocks.front());
        const float new_key[] = {0.5f, 0.5f};
        const float new_value[] = {100.0f, 200.0f};
        write_all_layers(suite, cache, request_b, 0, new_key, new_value);
        write_all_layers(suite, cache, request_b, 1, new_key, new_value);
        CPP_LLM_CHECK(suite, cache.commit_tokens(request_b, 2));

        std::vector<float> output;
        CPP_LLM_CHECK(
            suite, cache.attention(request_b, 0, {1.0f, 0.0f}, 1, &output));
        CPP_LLM_REQUIRE(suite, output.size() == 2);
        CPP_LLM_CHECK(suite, near(output[0], 100.0f));
        CPP_LLM_CHECK(suite, near(output[1], 200.0f));
        CPP_LLM_CHECK(suite, cache.release(request_b));
    });

    return suite.finish();
}
