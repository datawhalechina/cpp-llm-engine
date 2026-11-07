// Chapter 03：矩阵乘算子实现
// 流程：校验两个 Matrix 的形状，创建临时输出，再调用注入的计算后端。
// 提交：后端成功后才替换调用方输出；默认使用 CPU 参考后端。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#include "chapter03_matmul/matmul_op.h"
#include "chapter03_matmul/cpu_backend.h"

#include <stdexcept>
#include <utility>

namespace chapter03_matmul {

Matrix cpu_reference_matmul(const Matrix& lhs, const Matrix& rhs) {
    return MatMulOp().compute(lhs, rhs);
}

MatMulOp::MatMulOp() : backend_(&cpu_reference_backend()) {}

MatMulOp::MatMulOp(const cpp_llm_engine::core::MatmulBackend& backend) : backend_(&backend) {}

std::string_view MatMulOp::name() const { return "matmul"; }

cpp_llm_engine::core::BackendKind MatMulOp::backend_kind() const {
    return backend_->kind();
}

Matrix MatMulOp::compute(const Matrix& lhs, const Matrix& rhs) const {
    Matrix result;
    const auto status = try_compute(lhs, rhs, &result);
    if (!status.is_ok()) throw std::invalid_argument(status.message());
    return result;
}

cpp_llm_engine::core::Status MatMulOp::try_compute(
    const Matrix& lhs,
    const Matrix& rhs,
    Matrix* output) const {
    if (output == nullptr) {
        return cpp_llm_engine::core::Status::error("matmul output must not be null");
    }
    if (!lhs.ready() || !rhs.ready()) {
        return cpp_llm_engine::core::Status::error("matrix is not ready");
    }
    if (lhs.cols != rhs.rows) {
        return cpp_llm_engine::core::Status::error("matmul shape mismatch");
    }
    cpp_llm_engine::core::Shape output_shape{{lhs.rows, rhs.cols}};
    std::size_t output_elements = 0;
    const auto shape_status = output_shape.checked_numel(&output_elements);
    if (!shape_status.is_ok()) return shape_status;
    Matrix result(lhs.rows, rhs.cols, 0.0f);
    const cpp_llm_engine::core::MatmulRequest request{
        lhs.data.data(), rhs.data.data(), result.data.data(), lhs.rows, lhs.cols, rhs.cols};
    const auto status = backend_->compute(request);
    if (!status.is_ok()) return status;
    *output = std::move(result);
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace chapter03_matmul
