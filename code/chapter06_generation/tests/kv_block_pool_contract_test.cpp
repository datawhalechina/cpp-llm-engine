// Chapter 06：验证 KV 块池在预留、提交、裁剪和释放后的全局不变量。
// 职责：检查容量守恒、块独占、失败原子性以及释放后的复用。
// 验证入口：chapter06_kv_block_pool_contract_test 或带 contract 标签的 CTest。
#include "chapter06_generation/serving_preview.h"

#include "cpp_llm_engine/testing/test.h"

#include <cstddef>
#include <set>
#include <vector>

namespace {

void check_pool_invariants(
    cpp_llm_engine::testing::TestSuite& suite,
    const chapter06_generation::KvBlockPool& pool,
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
    cpp_llm_engine::testing::TestSuite suite("chapter06_kv_block_pool_contract");

    suite.run("lifecycle preserves capacity and ownership", [&] {
        chapter06_generation::KvBlockPool pool(5, 2);
        const std::vector<std::size_t> ids = {10, 11, 12};
        check_pool_invariants(suite, pool, ids);

        CPP_LLM_CHECK(suite, pool.reserve(10, 3));
        CPP_LLM_CHECK(suite, pool.commit_tokens(10, 3));
        check_pool_invariants(suite, pool, ids);

        CPP_LLM_CHECK(suite, pool.reserve(11, 4));
        CPP_LLM_CHECK(suite, pool.commit_tokens(11, 2));
        check_pool_invariants(suite, pool, ids);

        CPP_LLM_CHECK(suite, pool.trim_to_tokens(11, 2));
        check_pool_invariants(suite, pool, ids);

        const auto released_blocks = pool.physical_blocks(10);
        CPP_LLM_CHECK(suite, pool.release(10));
        CPP_LLM_CHECK(suite, pool.reserve(12, 3));
        const auto reused_blocks = pool.physical_blocks(12);
        CPP_LLM_REQUIRE(suite, !released_blocks.empty());
        CPP_LLM_REQUIRE(suite, !reused_blocks.empty());
        CPP_LLM_CHECK(suite, reused_blocks.front() == released_blocks.front());
        check_pool_invariants(suite, pool, ids);
    });

    suite.run("failed reserve does not partially allocate", [&] {
        chapter06_generation::KvBlockPool pool(2, 2);
        CPP_LLM_CHECK(suite, pool.reserve(20, 3));
        CPP_LLM_CHECK(suite, pool.commit_tokens(20, 3));
        const auto before = pool.physical_blocks(20);

        CPP_LLM_CHECK(suite, !pool.reserve(21, 1));
        CPP_LLM_CHECK(suite, !pool.contains(21));
        CPP_LLM_CHECK(suite, pool.physical_blocks(20) == before);
        check_pool_invariants(suite, pool, {20, 21});
    });

    suite.run("commit cannot exceed reservation or move backwards", [&] {
        chapter06_generation::KvBlockPool pool(3, 2);
        CPP_LLM_CHECK(suite, pool.reserve(30, 3));
        CPP_LLM_CHECK(suite, pool.commit_tokens(30, 2));
        CPP_LLM_CHECK(suite, !pool.commit_tokens(30, 5));
        CPP_LLM_CHECK(suite, !pool.commit_tokens(30, 1));
        CPP_LLM_CHECK(suite, pool.token_count(30) == 2);
        check_pool_invariants(suite, pool, {30});
    });

    return suite.finish();
}
