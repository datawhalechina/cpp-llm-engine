// 测试支持自检：Release 构建中 NDEBUG 必须被公共 CMake 配置撤销。
// 职责：验证命名用例、检查计数和非零失败返回值所依赖的基础入口。
// 验证入口：以 Release 模式构建 code/testing 并运行 CTest。
#include "cpp_llm_engine/testing/test.h"
#include "cpp_llm_engine/testing/fault_injector.h"

#include <stdexcept>

#ifdef NDEBUG
#error "cpp_llm_prepare_test must keep assertions enabled in Release builds"
#endif

int main() {
    cpp_llm_engine::testing::TestSuite suite("testing_self_test");
    suite.run("release checks and equality diagnostics remain active", [&] {
        CPP_LLM_REQUIRE(suite, true);
        CPP_LLM_CHECK_EQ(suite, 2 + 2, 4);
    });
    suite.run("expected exception is recognized", [&] {
        CPP_LLM_CHECK_THROWS(
            suite,
            std::runtime_error,
            throw std::runtime_error("expected"));
    });
    suite.run("one-shot fault fails only at the selected call", [&] {
        cpp_llm_engine::testing::FaultInjector faults;
        faults.arm("executor.prefill", 2);
        CPP_LLM_CHECK(suite, faults.armed("executor.prefill"));
        CPP_LLM_CHECK(suite, !faults.hit("executor.prefill"));
        CPP_LLM_CHECK(suite, faults.hit("executor.prefill"));
        CPP_LLM_CHECK(suite, !faults.hit("executor.prefill"));
        CPP_LLM_CHECK_EQ(suite, faults.calls("executor.prefill"), 3U);
        CPP_LLM_CHECK_EQ(suite, faults.failures("executor.prefill"), 1U);
    });
    suite.run("persistent fault fails from the selected call", [&] {
        cpp_llm_engine::testing::FaultInjector faults;
        faults.arm(
            "transport.write",
            2,
            cpp_llm_engine::testing::FaultMode::persistent);
        CPP_LLM_CHECK(suite, !faults.hit("transport.write"));
        CPP_LLM_CHECK(suite, faults.hit("transport.write"));
        CPP_LLM_CHECK(suite, faults.hit("transport.write"));
        CPP_LLM_CHECK_EQ(suite, faults.failures("transport.write"), 2U);
        faults.clear();
        CPP_LLM_CHECK(suite, !faults.armed("transport.write"));
    });
    suite.run("invalid fault configuration is rejected", [&] {
        cpp_llm_engine::testing::FaultInjector faults;
        CPP_LLM_CHECK_THROWS(suite, std::invalid_argument, faults.arm(""));
        CPP_LLM_CHECK_THROWS(
            suite, std::invalid_argument, faults.arm("executor.decode", 0));
    });
    return suite.finish();
}
