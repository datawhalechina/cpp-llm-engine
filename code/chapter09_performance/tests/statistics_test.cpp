// Chapter 09：验证重复样本的基础统计结果。
// 职责：锁定 count、min、max、mean 和 median 的定义。
// 验证入口：chapter09_statistics_test。
#include "chapter09_performance/statistics.h"

#include "cpp_llm_engine/testing/test.h"

#include <cmath>

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter09_statistics");

    suite.run("empty samples", [&] {
        const auto result = chapter09_performance::summarize({});
        CPP_LLM_CHECK(suite, result.count == 0);
        CPP_LLM_CHECK(suite, result.mean == 0.0);
    });

    suite.run("odd samples ignore input order", [&] {
        const auto result = chapter09_performance::summarize({9.0, 1.0, 5.0});
        CPP_LLM_CHECK(suite, result.count == 3);
        CPP_LLM_CHECK(suite, result.minimum == 1.0);
        CPP_LLM_CHECK(suite, result.maximum == 9.0);
        CPP_LLM_CHECK(suite, result.mean == 5.0);
        CPP_LLM_CHECK(suite, result.median == 5.0);
    });

    suite.run("even samples average middle values", [&] {
        const auto result = chapter09_performance::summarize({8.0, 2.0, 4.0, 6.0});
        CPP_LLM_CHECK(suite, std::abs(result.median - 5.0) < 1e-9);
    });

    return suite.finish();
}
