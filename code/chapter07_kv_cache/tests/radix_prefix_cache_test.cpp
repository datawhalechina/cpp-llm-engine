// Chapter 07：验证 Radix 最长前缀匹配、共享块所有权和私有后缀。
// 职责：固定 CPU 前缀缓存语义，不测试淘汰策略或调度优先级。
// 验证入口：chapter07_radix_prefix_cache_test。
#include "chapter07_kv_cache/radix_prefix_cache.h"

#include "cpp_llm_engine/testing/test.h"

#include <cstddef>
#include <vector>

namespace {

using chapter07_kv_cache::CpuPagedKvCache;
using chapter07_kv_cache::RadixPrefixCache;
using cpp_llm_engine::testing::TestSuite;

void append_token(
    TestSuite& suite,
    CpuPagedKvCache& cache,
    std::size_t request_id,
    std::size_t token,
    float payload) {
    const float key[] = {payload, payload + 0.25f};
    const float value[] = {payload + 10.0f, payload + 20.0f};
    CPP_LLM_CHECK(suite, cache.write_token(
        request_id, 0, token, key, value, 2));
}

}  // namespace

int main() {
    TestSuite suite("chapter07_radix_prefix_cache");

    suite.run("longest match shares aligned blocks and preserves payload", [&] {
        CpuPagedKvCache cache(1, 5, 2, 1, 2);
        RadixPrefixCache prefixes(&cache);
        constexpr std::size_t source = 100;
        constexpr std::size_t target = 200;

        CPP_LLM_REQUIRE(suite, cache.reserve(source, 5));
        for (std::size_t token = 0; token < 5; ++token) {
            append_token(suite, cache, source, token, static_cast<float>(token));
        }
        CPP_LLM_REQUIRE(suite, cache.commit_tokens(source, 5));
        const auto source_blocks = cache.physical_blocks(source);
        CPP_LLM_REQUIRE(suite, source_blocks.size() == 3);

        CPP_LLM_CHECK(suite, prefixes.publish({1, 2, 3, 4, 5}, source));
        CPP_LLM_CHECK(suite, prefixes.cached_blocks() == 2);
        CPP_LLM_CHECK(suite, cache.block_ref_count(source_blocks[0]) == 2);
        CPP_LLM_CHECK(suite, cache.block_ref_count(source_blocks[1]) == 2);

        // Re-publishing the same token path keeps the canonical blocks once.
        CPP_LLM_CHECK(suite, prefixes.publish({1, 2, 3, 4, 5}, source));
        CPP_LLM_CHECK(suite, cache.block_ref_count(source_blocks[0]) == 2);

        const auto match = prefixes.longest_prefix({1, 2, 3, 4, 9, 10});
        CPP_LLM_REQUIRE(suite, match.token_count == 4);
        CPP_LLM_CHECK(suite, match.physical_blocks.size() == 2);
        CPP_LLM_REQUIRE(suite, prefixes.attach(target, match));
        CPP_LLM_CHECK(suite, cache.token_count(target) == 4);
        CPP_LLM_CHECK(suite, cache.physical_blocks(target) == match.physical_blocks);
        CPP_LLM_CHECK(suite, cache.block_ref_count(source_blocks[0]) == 3);

        CPP_LLM_REQUIRE(suite, cache.release(source));
        CPP_LLM_CHECK(suite, cache.block_ref_count(source_blocks[0]) == 2);
        CPP_LLM_CHECK(suite, cache.value_data(target, 0, 0, 0)[0] == 10.0f);

        // A cache hit supplies immutable full blocks; generation gets a private tail.
        CPP_LLM_REQUIRE(suite, cache.reserve(target, 5));
        append_token(suite, cache, target, 4, 9.0f);
        CPP_LLM_REQUIRE(suite, cache.commit_tokens(target, 5));
        CPP_LLM_CHECK(suite, cache.block_count(target) == 3);
        const std::size_t private_tail = cache.physical_blocks(target).back();
        CPP_LLM_CHECK(suite, cache.block_ref_count(private_tail) == 1);

        CPP_LLM_REQUIRE(suite, cache.release(target));
        CPP_LLM_CHECK(suite, cache.block_ref_count(private_tail) == 0);
        CPP_LLM_CHECK(suite, cache.block_ref_count(source_blocks[0]) == 1);
        CPP_LLM_CHECK(suite, cache.free_blocks() == 3);

        prefixes.clear();
        CPP_LLM_CHECK(suite, prefixes.cached_blocks() == 0);
        CPP_LLM_CHECK(suite, cache.free_blocks() == cache.total_blocks());
    });

    suite.run("match stops at the last complete cached block", [&] {
        CpuPagedKvCache cache(1, 3, 2, 1, 2);
        RadixPrefixCache prefixes(&cache);
        CPP_LLM_REQUIRE(suite, cache.reserve(1, 3));
        for (std::size_t token = 0; token < 3; ++token) {
            append_token(suite, cache, 1, token, static_cast<float>(token));
        }
        CPP_LLM_REQUIRE(suite, cache.commit_tokens(1, 3));
        CPP_LLM_REQUIRE(suite, prefixes.publish({7, 8, 9}, 1));

        CPP_LLM_CHECK(suite, prefixes.longest_prefix({7}).token_count == 0);
        CPP_LLM_CHECK(suite, prefixes.longest_prefix({7, 8, 99}).token_count == 2);
        CPP_LLM_CHECK(suite, prefixes.longest_prefix({7, 99}).token_count == 0);
        CPP_LLM_CHECK(suite, !prefixes.attach(2, {}));
    });

    return suite.finish();
}
