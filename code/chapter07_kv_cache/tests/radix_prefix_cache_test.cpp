// Chapter 07：验证 Radix 最长前缀匹配、共享块所有权和私有后缀。
// 职责：固定 CPU 前缀缓存的匹配、所有权、统计和叶端淘汰语义。
// 验证入口：chapter07_radix_prefix_cache_test；调度优先级由 Chapter08 验证。
#include "chapter07_kv_cache/radix_prefix_cache.h"
#include "chapter07_kv_cache/paged_kv_cache.h"

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

        // 重复发布同一词元路径时，索引只保留一次规范块引用。
        CPP_LLM_CHECK(suite, prefixes.publish({1, 2, 3, 4, 5}, source));
        CPP_LLM_CHECK(suite, cache.block_ref_count(source_blocks[0]) == 2);

        const auto stats_before_estimate = prefixes.stats();
        const auto refs_before_estimate = cache.block_ref_count(source_blocks[0]);
        CPP_LLM_CHECK_EQ(
            suite, prefixes.estimate_prefix_tokens({1, 2, 3, 4, 9, 10}), 4U);
        CPP_LLM_CHECK_EQ(
            suite, prefixes.estimate_prefix_tokens({9, 9, 9, 9}), 0U);
        const auto stats_after_estimate = prefixes.stats();
        CPP_LLM_CHECK_EQ(
            suite, cache.block_ref_count(source_blocks[0]), refs_before_estimate);
        CPP_LLM_CHECK_EQ(suite, prefixes.cached_blocks(), 2U);
        CPP_LLM_CHECK_EQ(
            suite, stats_after_estimate.queries, stats_before_estimate.queries);
        CPP_LLM_CHECK_EQ(
            suite, stats_after_estimate.hits, stats_before_estimate.hits);
        CPP_LLM_CHECK_EQ(
            suite, stats_after_estimate.misses, stats_before_estimate.misses);
        CPP_LLM_CHECK_EQ(
            suite, stats_after_estimate.matched_tokens,
            stats_before_estimate.matched_tokens);

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

        // 命中提供只读完整块，后续生成使用请求私有尾块。
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

    suite.run("leaf LRU eviction preserves the reachable recent prefix", [&] {
        CpuPagedKvCache cache(1, 4, 2, 1, 2);
        RadixPrefixCache prefixes(&cache);
        for (std::size_t request = 1; request <= 2; ++request) {
            CPP_LLM_REQUIRE(suite, cache.reserve(request, 4));
            for (std::size_t token = 0; token < 4; ++token) {
                append_token(
                    suite, cache, request, token,
                    static_cast<float>(request * 10 + token));
            }
            CPP_LLM_REQUIRE(suite, cache.commit_tokens(request, 4));
            const std::vector<int> tokens = request == 1
                ? std::vector<int>{1, 2, 3, 4}
                : std::vector<int>{5, 6, 7, 8};
            CPP_LLM_REQUIRE(suite, prefixes.publish(tokens, request));
            CPP_LLM_REQUIRE(suite, cache.release(request));
        }
        CPP_LLM_CHECK_EQ(suite, cache.free_blocks(), 0U);
        CPP_LLM_CHECK_EQ(suite, prefixes.cached_blocks(), 4U);

        CPP_LLM_CHECK_EQ(
            suite, prefixes.longest_prefix({1, 2, 3, 4}).token_count, 4U);
        CPP_LLM_REQUIRE(suite, prefixes.evict_until_free(2));
        CPP_LLM_CHECK_EQ(suite, cache.free_blocks(), 2U);
        CPP_LLM_CHECK_EQ(
            suite, prefixes.longest_prefix({1, 2, 3, 4}).token_count, 4U);
        CPP_LLM_CHECK_EQ(
            suite, prefixes.longest_prefix({5, 6, 7, 8}).token_count, 0U);
        const auto stats = prefixes.stats();
        CPP_LLM_CHECK_EQ(suite, stats.queries, 3U);
        CPP_LLM_CHECK_EQ(suite, stats.hits, 2U);
        CPP_LLM_CHECK_EQ(suite, stats.misses, 1U);
        CPP_LLM_CHECK_EQ(suite, stats.matched_tokens, 8U);
        CPP_LLM_CHECK_EQ(suite, stats.evicted_blocks, 2U);
    });

    suite.run("active request references pin shared prefix blocks", [&] {
        CpuPagedKvCache cache(1, 2, 2, 1, 2);
        RadixPrefixCache prefixes(&cache);
        CPP_LLM_REQUIRE(suite, cache.reserve(1, 4));
        for (std::size_t token = 0; token < 4; ++token) {
            append_token(suite, cache, 1, token, static_cast<float>(token));
        }
        CPP_LLM_REQUIRE(suite, cache.commit_tokens(1, 4));
        CPP_LLM_REQUIRE(suite, prefixes.publish({1, 2, 3, 4}, 1));
        const auto match = prefixes.longest_prefix({1, 2, 3, 4});
        CPP_LLM_REQUIRE(suite, prefixes.attach(2, match));
        CPP_LLM_REQUIRE(suite, cache.release(1));

        CPP_LLM_CHECK(suite, !prefixes.evict_until_free(1));
        CPP_LLM_CHECK_EQ(suite, prefixes.cached_blocks(), 2U);
        CPP_LLM_REQUIRE(suite, cache.release(2));
        CPP_LLM_CHECK(suite, prefixes.evict_until_free(1));
        CPP_LLM_CHECK_EQ(suite, cache.free_blocks(), 1U);
        CPP_LLM_CHECK_EQ(suite, prefixes.cached_blocks(), 1U);
    });

    return suite.finish();
}
