// Chapter 03：Operator/backend smoke test
// 职责：验证 MatMulOp 名称、CPU backend 身份和最小 1x1 计算。
// 通过条件：公共 Operator 入口返回 2*3=6。
// 验证：chapter03_matmul_test 目标。
#include "chapter03_matmul/matmul_op.h"
#include "chapter03_matmul/cpu_backend.h"
#include "cpp_llm_engine/testing/test.h"

namespace {

void run_operator_smoke_test(cpp_llm_engine::testing::TestSuite& suite) {
    chapter03_matmul::Matrix lhs(1, 1, 2.0f);
    chapter03_matmul::Matrix rhs(1, 1, 3.0f);
    chapter03_matmul::MatMulOp op;
    CPP_LLM_CHECK(suite, op.name() == "matmul");
    CPP_LLM_CHECK(suite, op.backend_kind() == cpp_llm_engine::core::BackendKind::CPU);
    const chapter03_matmul::Matrix out = op.compute(lhs, rhs);
    CPP_LLM_CHECK_EQ(suite, out.rows, 1U);
    CPP_LLM_CHECK_EQ(suite, out.cols, 1U);
    CPP_LLM_CHECK_EQ(suite, out.at(0, 0), 6.0f);
}

}  // namespace

void chapter03_matmul_run_operator_smoke_test(
    cpp_llm_engine::testing::TestSuite& suite) {
    run_operator_smoke_test(suite);
}
