// CUDA Backend：实现 row-major MatMul 到 cuBLAS column-major SGEMM 的映射。
// 契约：计算 C^T=B^T*A^T，不转置或复制输入；每次调用显式设置 stream。
// 验证：cpp_llm_engine_cuda_cublas_matmul_test 与 CPU 三重循环对照。
#include "cpp_llm_engine/cuda/cublas_matmul_backend.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <limits>
#include <utility>

namespace cpp_llm_engine::cuda {
namespace {

bool fits_cublas_int(std::size_t value) {
    return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

}  // namespace

CublasMatmulBackend::~CublasMatmulBackend() { release_noexcept(); }

CublasMatmulBackend::CublasMatmulBackend(CublasMatmulBackend&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {}

CublasMatmulBackend& CublasMatmulBackend::operator=(
    CublasMatmulBackend&& other) noexcept {
    if (this == &other) return *this;
    release_noexcept();
    handle_ = std::exchange(other.handle_, nullptr);
    return *this;
}

CublasStatus CublasMatmulBackend::create(CublasMatmulBackend* output) {
    if (output == nullptr) {
        return CublasStatus::from(
            CUBLAS_STATUS_INVALID_VALUE, "CublasMatmulBackend::create output");
    }
    CublasMatmulBackend replacement;
    const CublasStatus status = CublasStatus::from(
        cublasCreate(&replacement.handle_), "cublasCreate");
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return CublasStatus::success();
}

core::BackendKind CublasMatmulBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CublasMatmulBackend::name() const {
    return "cublas-sgemm";
}

CublasStatus CublasMatmulBackend::compute(
    const core::MatmulRequest& request,
    const CudaStream& stream) const {
    if (!valid()) {
        return CublasStatus::from(
            CUBLAS_STATUS_NOT_INITIALIZED, "CublasMatmulBackend::compute handle");
    }
    if (!stream.valid()) {
        return CublasStatus::from(
            CUBLAS_STATUS_INVALID_VALUE, "CublasMatmulBackend::compute stream");
    }
    if (request.lhs == nullptr || request.rhs == nullptr || request.output == nullptr) {
        return CublasStatus::from(
            CUBLAS_STATUS_INVALID_VALUE, "CublasMatmulBackend::compute pointers");
    }
    if (request.rows == 0 || request.inner == 0 || request.cols == 0 ||
        !fits_cublas_int(request.rows) || !fits_cublas_int(request.inner) ||
        !fits_cublas_int(request.cols)) {
        return CublasStatus::from(
            CUBLAS_STATUS_INVALID_VALUE, "CublasMatmulBackend::compute dimensions");
    }

    CublasStatus status = CublasStatus::from(
        cublasSetStream(handle_, stream.native_handle()), "cublasSetStream");
    if (!status.is_ok()) return status;

    const int rows = static_cast<int>(request.rows);
    const int inner = static_cast<int>(request.inner);
    const int cols = static_cast<int>(request.cols);
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    return CublasStatus::from(
        cublasSgemm(
            handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            cols, rows, inner,
            &alpha,
            request.rhs, cols,
            request.lhs, inner,
            &beta,
            request.output, cols),
        "cublasSgemm row-major");
}

CublasStatus CublasMatmulBackend::compute_rhs_transposed(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t rows,
    std::size_t inner,
    std::size_t cols,
    const CudaStream& stream) const {
    if (!valid()) {
        return CublasStatus::from(
            CUBLAS_STATUS_NOT_INITIALIZED,
            "CublasMatmulBackend::compute_rhs_transposed handle");
    }
    if (!stream.valid() || lhs == nullptr || rhs == nullptr || output == nullptr ||
        rows == 0 || inner == 0 || cols == 0 || !fits_cublas_int(rows) ||
        !fits_cublas_int(inner) || !fits_cublas_int(cols)) {
        return CublasStatus::from(
            CUBLAS_STATUS_INVALID_VALUE,
            "CublasMatmulBackend::compute_rhs_transposed input");
    }

    CublasStatus status = CublasStatus::from(
        cublasSetStream(handle_, stream.native_handle()), "cublasSetStream");
    if (!status.is_ok()) return status;

    const int batch_rows = static_cast<int>(rows);
    const int input_width = static_cast<int>(inner);
    const int output_width = static_cast<int>(cols);
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    // Row-major output [B,N] is column-major [N,B]. The stored rhs [N,K]
    // appears as [K,N], so OP_T recovers [N,K] without a transpose buffer.
    return CublasStatus::from(
        cublasSgemm(
            handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            output_width, batch_rows, input_width,
            &alpha,
            rhs, input_width,
            lhs, input_width,
            &beta,
            output, output_width),
        "cublasSgemm row-major rhs-transposed");
}

CublasStatus CublasMatmulBackend::reset() {
    if (!valid()) return CublasStatus::success();
    const cublasHandle_t handle = handle_;
    handle_ = nullptr;
    return CublasStatus::from(cublasDestroy(handle), "cublasDestroy");
}

bool CublasMatmulBackend::valid() const { return handle_ != nullptr; }

void CublasMatmulBackend::release_noexcept() noexcept {
    if (handle_ != nullptr) (void)cublasDestroy(handle_);
    handle_ = nullptr;
}

}  // namespace cpp_llm_engine::cuda
