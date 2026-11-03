// Chapter 06：Request 本地 KV Cache 命名契约测试。
// 职责：验证 token history 的追加顺序和 clear 行为。
// 验证入口：Chapter06 CTest 或带 unit、cache 标签的 CTest。
#include "chapter06_generation/kv_cache.h"

#include "cpp_llm_engine/testing/test.h"

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter06_request_kv_cache");

    suite.run("append preserves order and clear resets history", [&] {
        chapter06_generation::KVCache cache;
        cache.append(1);
        cache.append(2);
        CPP_LLM_CHECK(suite, cache.size() == 2);
        CPP_LLM_CHECK(suite, cache.tokens() == chapter06_generation::TokenSequence({1, 2}));
        cache.clear();
        CPP_LLM_CHECK(suite, cache.empty());
        CPP_LLM_CHECK(suite, cache.tokens().empty());
    });

    return suite.finish();
}
