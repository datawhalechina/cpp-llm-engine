// 测试支持：提供不受 NDEBUG 影响的命名用例和检查结果。
// 职责：让教学测试报告失败用例、表达式和源码位置，不替代 CTest。
// 用法：创建 TestSuite，以 run() 注册用例，并从 main() 返回 finish()。
#pragma once

#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace cpp_llm_engine::testing {

class TestSuite final {
public:
    explicit TestSuite(std::string name) : name_(std::move(name)) {}

    template <typename TestCase>
    void run(const std::string& case_name, TestCase&& test_case) {
        ++case_count_;
        current_case_ = case_name;
        try {
            test_case();
        } catch (const AbortCase&) {
            // require() 已经记录失败；仅终止当前用例，继续执行后续用例。
        } catch (const std::exception& error) {
            fail("unexpected exception", __FILE__, __LINE__, error.what());
        } catch (...) {
            fail("unexpected exception", __FILE__, __LINE__, "unknown exception");
        }
    }

    void check(bool condition, const char* expression, const char* file, int line) {
        ++check_count_;
        if (!condition) fail(expression, file, line, "");
    }

    void require(bool condition, const char* expression, const char* file, int line) {
        ++check_count_;
        if (condition) return;
        fail(expression, file, line, "required condition is false");
        throw AbortCase{};
    }

    template <typename Left, typename Right>
    void check_equal(
        const Left& left,
        const Right& right,
        const char* left_expression,
        const char* right_expression,
        const char* file,
        int line) {
        ++check_count_;
        if (left == right) return;
        std::ostringstream detail;
        detail << left_expression << '=' << left << ", "
               << right_expression << '=' << right;
        fail("values are equal", file, line, detail.str());
    }

    template <typename Exception, typename Function>
    void check_throws(
        Function&& function,
        const char* expression,
        const char* exception_name,
        const char* file,
        int line) {
        ++check_count_;
        try {
            function();
        } catch (const Exception&) {
            return;
        } catch (const std::exception& error) {
            fail(expression, file, line, std::string("unexpected exception: ") + error.what());
            return;
        } catch (...) {
            fail(expression, file, line, "unexpected non-standard exception");
            return;
        }
        fail(expression, file, line, std::string("expected exception: ") + exception_name);
    }

    int finish() const {
        std::cout << "[test] suite=" << name_
                  << " cases=" << case_count_
                  << " checks=" << check_count_
                  << " failures=" << failure_count_ << '\n';
        return failure_count_ == 0 ? 0 : 1;
    }

private:
    struct AbortCase final {};

    void fail(
        const char* expression,
        const char* file,
        int line,
        const std::string& detail) {
        ++failure_count_;
        std::cerr << file << ':' << line << ": case '" << current_case_
                  << "' failed: " << expression;
        if (!detail.empty()) std::cerr << " (" << detail << ')';
        std::cerr << '\n';
    }

    std::string name_;
    std::string current_case_;
    std::size_t case_count_ = 0;
    std::size_t check_count_ = 0;
    std::size_t failure_count_ = 0;
};

}  // namespace cpp_llm_engine::testing

#define CPP_LLM_CHECK(suite, expression) \
    (suite).check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

#define CPP_LLM_REQUIRE(suite, expression) \
    (suite).require(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

#define CPP_LLM_CHECK_EQ(suite, left, right) \
    (suite).check_equal((left), (right), #left, #right, __FILE__, __LINE__)

#define CPP_LLM_CHECK_THROWS(suite, exception_type, expression) \
    (suite).check_throws<exception_type>( \
        [&] { expression; }, \
        #expression, \
        #exception_type, \
        __FILE__, \
        __LINE__)
