// Chapter 03：MatMul 算子接口
// 职责：把 Matrix 输入转换为公共 MatmulRequest，并注入可替换 backend。
// 契约：try_compute 成功后提交输出、失败时保留输出；compute 失败时抛出明确错误。
// 验证：chapter03_matmul_test.cpp 和 operator_smoke_test.cpp。
#pragma once

#include "chapter03_matmul/operator.h"
#include "cpp_llm_engine/core/matmul_contract.h"

namespace chapter03_matmul {

class MatMulOp final : public Operator {
public:
    MatMulOp();
    explicit MatMulOp(const cpp_llm_engine::core::MatmulBackend& backend);

    std::string_view name() const override;
    cpp_llm_engine::core::BackendKind backend_kind() const;
    cpp_llm_engine::core::Status try_compute(
        const Matrix& lhs, const Matrix& rhs, Matrix* output) const;
    Matrix compute(const Matrix& lhs, const Matrix& rhs) const override;

private:
    const cpp_llm_engine::core::MatmulBackend* backend_;
};

Matrix cpu_reference_matmul(const Matrix& lhs, const Matrix& rhs);

}  // namespace chapter03_matmul
