// Chapter 07：验证分页 MLA 的压缩载荷、提交原子性和块复用隔离。
// 职责：固定 latent K/V 与 RoPE key 的分页数据契约。
// 验证入口：chapter07_paged_mla_cache_test。
#include "chapter07_kv_cache/paged_mla_cache.h"

#include "cpp_llm_engine/testing/test.h"

#include <string>
#include <vector>

namespace {

using chapter07_kv_cache::CpuPagedMlaCache;
using cpp_llm_engine::testing::TestSuite;

void write_all_layers(
    TestSuite& suite,
    CpuPagedMlaCache& cache,
    std::size_t request_id,
    std::size_t token,
    const float* latent,
    const float* key_rope) {
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        CPP_LLM_CHECK(suite, cache.write_token(
            request_id,
            layer,
            token,
            latent,
            cache.latent_dimension(),
            key_rope,
            cache.rope_dimension()));
    }
}

}  // namespace

int main() {
    TestSuite suite("chapter07_paged_mla_cache");

    suite.run("cross block gather preserves logical token order", [&] {
        CpuPagedMlaCache cache(2, 4, 2, 3, 2);
        CPP_LLM_CHECK(
            suite,
            cache.payload_bytes() == 2 * 4 * 2 * (3 + 2) * sizeof(float));
        constexpr std::size_t request_id = 70;
        const std::vector<float> latent{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f};
        const std::vector<float> rope{
            10.0f, 11.0f,
            12.0f, 13.0f,
            14.0f, 15.0f};
        CPP_LLM_CHECK(suite, cache.reserve(request_id, 3));
        for (std::size_t token = 0; token < 3; ++token) {
            write_all_layers(
                suite,
                cache,
                request_id,
                token,
                latent.data() + token * 3,
                rope.data() + token * 2);
        }
        CPP_LLM_CHECK(suite, cache.commit_tokens(request_id, 3));
        CPP_LLM_CHECK(suite, cache.block_count(request_id) == 2);
        CPP_LLM_CHECK(
            suite,
            cache.locate(request_id, 0).block != cache.locate(request_id, 2).block);

        std::vector<float> gathered_latent;
        std::vector<float> gathered_rope;
        CPP_LLM_CHECK(suite, cache.gather_layer(
            request_id, 1, &gathered_latent, &gathered_rope));
        CPP_LLM_CHECK(suite, gathered_latent == latent);
        CPP_LLM_CHECK(suite, gathered_rope == rope);
    });

    suite.run("commit requires every layer and batch is atomic", [&] {
        CpuPagedMlaCache cache(2, 4, 2, 2, 1);
        const float latent[] = {1.0f, 2.0f};
        const float rope[] = {3.0f};
        CPP_LLM_CHECK(suite, cache.reserve(71, 1));
        CPP_LLM_CHECK(suite, cache.reserve(72, 1));
        write_all_layers(suite, cache, 71, 0, latent, rope);
        CPP_LLM_CHECK(suite, cache.write_token(72, 0, 0, latent, 2, rope, 1));

        std::string error;
        CPP_LLM_CHECK(suite, !cache.commit_batch({{71, 1}, {72, 1}}, &error));
        CPP_LLM_CHECK(suite, error.find("every layer") != std::string::npos);
        CPP_LLM_CHECK(suite, cache.token_count(71) == 0);
        CPP_LLM_CHECK(suite, cache.token_count(72) == 0);

        CPP_LLM_CHECK(suite, cache.write_token(72, 1, 0, latent, 2, rope, 1));
        CPP_LLM_CHECK(suite, cache.commit_batch({{71, 1}, {72, 1}}, &error));
        CPP_LLM_CHECK(suite, cache.token_count(71) == 1);
        CPP_LLM_CHECK(suite, cache.token_count(72) == 1);
        CPP_LLM_CHECK(
            suite,
            !cache.write_token(71, 0, 0, latent, 2, rope, 1, &error));
        CPP_LLM_CHECK(suite, error.find("immutable") != std::string::npos);
    });

    suite.run("trim clears write markers before physical block reuse", [&] {
        CpuPagedMlaCache cache(2, 2, 1, 2, 1);
        const float latent[] = {1.0f, 2.0f};
        const float rope[] = {3.0f};
        CPP_LLM_CHECK(suite, cache.reserve(73, 2));
        write_all_layers(suite, cache, 73, 0, latent, rope);
        CPP_LLM_CHECK(suite, cache.commit_tokens(73, 1));
        CPP_LLM_CHECK(suite, cache.write_token(73, 0, 1, latent, 2, rope, 1));
        const std::size_t released_block = cache.physical_blocks(73).back();
        CPP_LLM_CHECK(suite, cache.trim_to_tokens(73, 1));

        CPP_LLM_CHECK(suite, cache.reserve(74, 1));
        CPP_LLM_CHECK(
            suite,
            cache.physical_blocks(74).front() == released_block);
        CPP_LLM_CHECK(suite, cache.write_token(74, 1, 0, latent, 2, rope, 1));
        std::string error;
        CPP_LLM_CHECK(suite, !cache.commit_tokens(74, 1, &error));
        CPP_LLM_CHECK(suite, error.find("every layer") != std::string::npos);
        CPP_LLM_CHECK(suite, cache.write_token(74, 0, 0, latent, 2, rope, 1));
        CPP_LLM_CHECK(suite, cache.commit_tokens(74, 1));
        CPP_LLM_CHECK(suite, cache.release(74));
    });

    suite.run("invalid writes preserve request state", [&] {
        CpuPagedMlaCache cache(2, 2, 2, 2, 1);
        const float latent[] = {1.0f, 2.0f};
        const float rope[] = {3.0f};
        std::string error;
        CPP_LLM_CHECK(suite, cache.reserve(75, 1));
        CPP_LLM_CHECK(
            suite,
            !cache.write_token(75, 0, 0, latent, 1, rope, 1, &error));
        CPP_LLM_CHECK(suite, error.find("shape") != std::string::npos);
        CPP_LLM_CHECK(
            suite,
            !cache.write_token(75, 2, 0, latent, 2, rope, 1, &error));
        CPP_LLM_CHECK(suite, error.find("layer") != std::string::npos);
        CPP_LLM_CHECK(suite, cache.token_count(75) == 0);
        CPP_LLM_CHECK(suite, cache.write_token(75, 0, 0, latent, 2, rope, 1));
        CPP_LLM_CHECK(suite, cache.write_token(75, 1, 0, latent, 2, rope, 1));
        CPP_LLM_CHECK(suite, cache.commit_tokens(75, 1));
    });

    return suite.finish();
}
