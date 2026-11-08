// CUDA Backend：声明拥有独立 cuBLAS handle 的 row-major MatMul backend。
// 职责：复用 core MatmulRequest，并在调用时把 handle 绑定到显式 stream。
// 验证：多 shape CPU golden、非法输入、移动所有权和 event benchmark。
#pragma once

#include "cpp_llm_engine/core/backend.h"
#include "cpp_llm_engine/core/matmul_contract.h"
#include "cpp_llm_engine/cuda/cublas_status.h"

#include <cublas_v2.h>

#include <string_view>

namespace cpp_llm_engine::cuda {

class CudaStream;

class CublasMatmulBackend final : public core::Backend {
public:
    CublasMatmulBackend() = default;
    ~CublasMatmulBackend();

    CublasMatmulBackend(const CublasMatmulBackend&) = delete;
    CublasMatmulBackend& operator=(const CublasMatmulBackend&) = delete;
    CublasMatmulBackend(CublasMatmulBackend&& other) noexcept;
    CublasMatmulBackend& operator=(CublasMatmulBackend&& other) noexcept;

    static CublasStatus create(CublasMatmulBackend* output);

    core::BackendKind kind() const override;
    std::string_view name() const override;
    CublasStatus compute(
        const core::MatmulRequest& request,
        const CudaStream& stream) const;
    CublasStatus compute_rhs_transposed(
        const float* lhs,
        const float* rhs,
        float* output,
        std::size_t rows,
        std::size_t inner,
        std::size_t cols,
        const CudaStream& stream) const;
    CublasStatus reset();
    bool valid() const;

private:
    void release_noexcept() noexcept;

    cublasHandle_t handle_ = nullptr;
};

}  // namespace cpp_llm_engine::cuda
