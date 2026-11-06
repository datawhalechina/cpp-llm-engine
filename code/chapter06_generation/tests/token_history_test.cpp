// Chapter 06：Request 词元历史契约测试。
// 职责：验证词元历史的追加顺序和 clear 行为。
// 验证入口：chapter06_token_history_test。
#include "chapter06_generation/token_history.h"

#include "cpp_llm_engine/testing/test.h"

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter06_token_history");

    suite.run("append preserves order and clear resets history", [&] {
        chapter06_generation::TokenHistory history;
        history.append(1);
        history.append(2);
        CPP_LLM_CHECK(suite, history.size() == 2);
        CPP_LLM_CHECK(
            suite, history.tokens() == chapter06_generation::TokenSequence({1, 2}));
        history.clear();
        CPP_LLM_CHECK(suite, history.empty());
        CPP_LLM_CHECK(suite, history.tokens().empty());
    });

    return suite.finish();
}
