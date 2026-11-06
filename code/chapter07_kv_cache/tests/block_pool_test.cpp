// Chapter 07：验证块池容量、页表定位和事务失败原子性。
// 职责：固定控制面契约，不涉及 K/V 数值和调度策略。
// 验证入口：chapter07_block_pool_test。
#include "chapter07_kv_cache/block_table.h"

#include "cpp_llm_engine/testing/test.h"

#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

void check_pool_invariants(
    cpp_llm_engine::testing::TestSuite& suite,
    const chapter07_kv_cache::KvBlockPool& pool,
    const std::vector<std::size_t>& request_ids) {
    CPP_LLM_CHECK(suite, pool.free_blocks() + pool.used_blocks() == pool.total_blocks());
    std::set<std::size_t> owned_blocks;
    std::size_t counted_blocks = 0;
    for (const std::size_t request_id : request_ids) {
        if (!pool.contains(request_id)) continue;
        const auto blocks = pool.physical_blocks(request_id);
        counted_blocks += blocks.size();
        CPP_LLM_CHECK(suite, pool.block_count(request_id) == blocks.size());
        CPP_LLM_CHECK(suite, pool.token_count(request_id) <= pool.capacity_tokens(request_id));
        for (const std::size_t block : blocks) {
            CPP_LLM_CHECK(suite, block < pool.total_blocks());
            CPP_LLM_CHECK(suite, owned_blocks.insert(block).second);
        }
    }
    CPP_LLM_CHECK(suite, counted_blocks == pool.used_blocks());
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter07_block_pool");
    using chapter07_kv_cache::KvBlockPool;

    suite.run("invalid block dimensions are rejected", [&] {
        bool invalid_table = false;
        bool invalid_pool = false;
        bool overflow_pool = false;
        try {
            chapter07_kv_cache::BlockTable table(0);
        } catch (const std::invalid_argument&) {
            invalid_table = true;
        }
        try {
            KvBlockPool pool(0, 2);
        } catch (const std::invalid_argument&) {
            invalid_pool = true;
        }
        try {
            KvBlockPool pool(2, std::numeric_limits<std::size_t>::max());
        } catch (const std::overflow_error&) {
            overflow_pool = true;
        }
        CPP_LLM_CHECK(suite, invalid_table);
        CPP_LLM_CHECK(suite, invalid_pool);
        CPP_LLM_CHECK(suite, overflow_pool);
    });

    suite.run("reserve commit locate and trim", [&] {
        KvBlockPool pool(4, 2);
        CPP_LLM_CHECK(suite, pool.reserve(10, 3));
        CPP_LLM_CHECK(suite, pool.block_count(10) == 2);
        CPP_LLM_CHECK(suite, pool.capacity_tokens(10) == 4);
        CPP_LLM_CHECK(suite, pool.token_count(10) == 0);
        CPP_LLM_CHECK(suite, pool.commit_tokens(10, 3));
        CPP_LLM_CHECK(suite, pool.locate(10, 2).slot == 0);
        CPP_LLM_CHECK(suite, pool.trim_to_tokens(10, 2));
        CPP_LLM_CHECK(suite, pool.block_count(10) == 1);
        CPP_LLM_CHECK(suite, pool.free_blocks() == 3);
    });

    suite.run("capacity failure leaves ownership unchanged", [&] {
        KvBlockPool pool(2, 2);
        CPP_LLM_CHECK(suite, pool.reserve(20, 3));
        CPP_LLM_CHECK(suite, pool.commit_tokens(20, 3));
        const auto owner_blocks = pool.physical_blocks(20);
        CPP_LLM_CHECK(suite, !pool.reserve(21, 1));
        CPP_LLM_CHECK(suite, !pool.contains(21));
        CPP_LLM_CHECK(suite, pool.free_blocks() == 0);
        CPP_LLM_CHECK(suite, pool.physical_blocks(20) == owner_blocks);
        check_pool_invariants(suite, pool, {20, 21});
    });

    suite.run("commit cannot exceed reservation or move backwards", [&] {
        KvBlockPool pool(3, 2);
        CPP_LLM_CHECK(suite, pool.reserve(22, 3));
        CPP_LLM_CHECK(suite, pool.commit_tokens(22, 2));
        CPP_LLM_CHECK(suite, !pool.commit_tokens(22, 5));
        CPP_LLM_CHECK(suite, !pool.commit_tokens(22, 1));
        CPP_LLM_CHECK(suite, pool.token_count(22) == 2);
        check_pool_invariants(suite, pool, {22});
    });

    suite.run("failed step trims reserved growth to committed tokens", [&] {
        KvBlockPool pool(4, 2);
        CPP_LLM_CHECK(suite, pool.reserve(25, 2));
        CPP_LLM_CHECK(suite, pool.commit_tokens(25, 2));
        const auto committed_blocks = pool.physical_blocks(25);
        CPP_LLM_CHECK(suite, pool.reserve(25, 5));
        CPP_LLM_CHECK(suite, pool.block_count(25) == 3);
        CPP_LLM_CHECK(suite, pool.token_count(25) == 2);

        CPP_LLM_CHECK(suite, pool.trim_to_tokens(25, 2));
        CPP_LLM_CHECK(suite, pool.block_count(25) == 1);
        CPP_LLM_CHECK(suite, pool.token_count(25) == 2);
        CPP_LLM_CHECK(suite, pool.physical_blocks(25) == committed_blocks);
        CPP_LLM_CHECK(suite, pool.free_blocks() == 3);
    });

    suite.run("release makes blocks reusable", [&] {
        KvBlockPool pool(2, 2);
        CPP_LLM_CHECK(suite, pool.reserve(30, 2));
        CPP_LLM_CHECK(suite, pool.commit_tokens(30, 2));
        const auto released = pool.physical_blocks(30);
        CPP_LLM_CHECK(suite, pool.release(30));
        CPP_LLM_CHECK(suite, pool.reserve(31, 2));
        CPP_LLM_REQUIRE(suite, !released.empty());
        CPP_LLM_CHECK(suite, pool.physical_blocks(31).front() == released.front());
    });

    suite.run("uncommitted locations cannot be read", [&] {
        KvBlockPool pool(2, 2);
        CPP_LLM_CHECK(suite, pool.reserve(40, 1));
        CPP_LLM_CHECK(suite, pool.locate_reserved(40, 0).slot == 0);
        bool rejected = false;
        try {
            static_cast<void>(pool.locate(40, 0));
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
    });

    return suite.finish();
}
