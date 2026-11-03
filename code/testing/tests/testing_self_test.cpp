// 测试支持自检：Release 构建中 NDEBUG 必须被公共 CMake 配置撤销。
// 职责：验证命名用例、检查计数和非零失败返回值所依赖的基础入口。
// 验证入口：以 Release 模式构建 code/testing 并运行 CTest。
#include "cpp_llm_engine/testing/test.h"

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
    return suite.finish();
}
