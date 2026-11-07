// Chapter 03：矩阵乘算子接口
// 输入输出：把两个 Matrix 转换为 MatmulRequest，成功后提交新的输出 Matrix。
// 生命周期：算子不拥有注入的计算后端，后端必须比算子存活更久；失败时保留原输出。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#pragma once

#include "chapter03_matmul/operator.h"
#include "cpp_llm_engine/core/matmul_contract.h"

namespace chapter03_matmul {

class MatMulOp final : public Operator {
public:
    MatMulOp();
    explicit MatMulOp(const cpp_llm_engine::core::MatmulBackend& backend);
    MatMulOp(cpp_llm_engine::core::MatmulBackend&& backend) = delete;

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
