// Chapter 07：声明轻量级 benchmark 测试用例和汇总结构。
// 职责：为本章示例提供不依赖外部测试库的确定性断言入口。
// 验证：chapter07_benchmark_test 使用这些结构报告通过和失败数量。
#pragma once

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string_view>

namespace chapter07_benchmark {

struct TestCase {
    std::string_view name;
    bool (*fn)();
};

struct TestSummary {
    std::size_t passed = 0;
    std::size_t failed = 0;
};

inline TestSummary run_test_suite(std::initializer_list<TestCase> tests) {
    TestSummary summary;
    for (const auto& test : tests) {
        const bool ok = test.fn();
        std::cout << "[test] " << test.name << ": " << (ok ? "PASS" : "FAIL") << '\n';
        if (ok) {
            ++summary.passed;
        } else {
            ++summary.failed;
        }
    }
    std::cout << "[test] summary: passed=" << summary.passed << ", failed=" << summary.failed << '\n';
    return summary;
}

}  // namespace chapter07_benchmark
