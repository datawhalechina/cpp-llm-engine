// Chapter 03：MatMul 数值与边界测试
// 职责：验证矩阵结果、shape mismatch、空请求、越界访问和超大 shape。
// 通过条件：命名数值、边界和失败原子性检查全部通过。
// 验证：bash code/chapter03_matmul/scripts/run.sh。
#include "chapter03_matmul/matmul_op.h"
#include "chapter03_matmul/cpu_kernel.h"
#include "cpp_llm_engine/testing/test.h"

#include <limits>
#include <stdexcept>

void chapter03_matmul_run_operator_smoke_test(
    cpp_llm_engine::testing::TestSuite& suite);

namespace {

class FailingCpuBackend final : public cpp_llm_engine::core::MatmulBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override {
        return cpp_llm_engine::core::BackendKind::CPU;
    }

    std::string_view name() const override { return "failing-cpu"; }

    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::MatmulRequest&) const override {
        return cpp_llm_engine::core::Status::error("injected backend failure");
    }
};

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter03_matmul");
    suite.run("operator exposes CPU backend and computes one element", [&] {
        chapter03_matmul_run_operator_smoke_test(suite);
    });

    suite.run("CPU reference matches fixed matrix result and contract", [&] {
        chapter03_matmul::Matrix lhs(2, 3);
        lhs.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        chapter03_matmul::Matrix rhs(3, 2);
        rhs.data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

        const auto out = chapter03_matmul::cpu_reference_matmul(lhs, rhs);
        CPP_LLM_CHECK_EQ(suite, out.rows, 2U);
        CPP_LLM_CHECK_EQ(suite, out.cols, 2U);
        CPP_LLM_CHECK_EQ(suite, out.at(0, 0), 58.0f);
        CPP_LLM_CHECK_EQ(suite, out.at(0, 1), 64.0f);
        CPP_LLM_CHECK_EQ(suite, out.at(1, 0), 139.0f);
        CPP_LLM_CHECK_EQ(suite, out.at(1, 1), 154.0f);
        const auto contract = out.contract();
        CPP_LLM_CHECK(
            suite, contract.shape.dimensions == std::vector<std::size_t>({2, 2}));
        CPP_LLM_CHECK(suite, contract.strides == std::vector<std::size_t>({2, 1}));
        CPP_LLM_CHECK(suite, contract.validate_element_view(sizeof(float)).is_ok());
    });

    suite.run("validation and backend failure preserve previous output", [&] {
        chapter03_matmul::Matrix lhs(2, 3, 1.0f);
        chapter03_matmul::Matrix rhs(3, 2, 1.0f);
        chapter03_matmul::MatMulOp op;
        chapter03_matmul::Matrix output;
        CPP_LLM_REQUIRE(suite, op.try_compute(lhs, rhs, &output).is_ok());
        const auto previous = output.data;
        const auto mismatch = op.try_compute(
            chapter03_matmul::Matrix(2, 3),
            chapter03_matmul::Matrix(2, 2),
            &output);
        CPP_LLM_CHECK(suite, !mismatch.is_ok());
        CPP_LLM_CHECK(suite, mismatch.message() == "matmul shape mismatch");
        CPP_LLM_CHECK(suite, output.data == previous);

        const FailingCpuBackend backend;
        const chapter03_matmul::MatMulOp failing_op(backend);
        const auto failure = failing_op.try_compute(lhs, rhs, &output);
        CPP_LLM_CHECK(suite, !failure.is_ok());
        CPP_LLM_CHECK(suite, failure.message() == "injected backend failure");
        CPP_LLM_CHECK(suite, output.data == previous);
        CPP_LLM_CHECK(suite, !op.try_compute(lhs, rhs, nullptr).is_ok());
    });

    suite.run("invalid request index and oversized shape are rejected", [&] {
        CPP_LLM_CHECK_THROWS(
            suite,
            std::invalid_argument,
            chapter03_matmul::cpu_reference_matmul(
                chapter03_matmul::Matrix(2, 3), chapter03_matmul::Matrix(2, 2)));
        CPP_LLM_CHECK(
            suite, !chapter03_matmul::cpu_reference_matmul_kernel({}).is_ok());
        chapter03_matmul::Matrix one(1, 1, 9.0f);
        CPP_LLM_CHECK_THROWS(suite, std::out_of_range, one.at(1, 0));
        CPP_LLM_CHECK_THROWS(
            suite,
            std::length_error,
            chapter03_matmul::Matrix(std::numeric_limits<std::size_t>::max(), 2));
    });
    return suite.finish();
}
