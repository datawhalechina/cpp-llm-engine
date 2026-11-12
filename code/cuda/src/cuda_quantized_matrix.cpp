// CUDA 执行层：实现 packed 量化矩阵的 device storage 所有权。
// 契约：创建时使用 core block contract 校验 shape，上传必须匹配完整 payload。
// 验证：cpp_llm_engine_cuda_quantized_linear_test。
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"

#include "cpp_llm_engine/core/quantization_contract.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <utility>

namespace cpp_llm_engine::cuda {

CudaQuantizedMatrix::CudaQuantizedMatrix(CudaQuantizedMatrix&& other) noexcept
    : storage_(std::move(other.storage_)),
      rows_(std::exchange(other.rows_, 0)),
      cols_(std::exchange(other.cols_, 0)),
      format_(std::exchange(other.format_, core::WeightFormat::Unknown)) {}

CudaQuantizedMatrix& CudaQuantizedMatrix::operator=(
    CudaQuantizedMatrix&& other) noexcept {
    if (this == &other) return *this;
    storage_ = std::move(other.storage_);
    rows_ = std::exchange(other.rows_, 0);
    cols_ = std::exchange(other.cols_, 0);
    format_ = std::exchange(other.format_, core::WeightFormat::Unknown);
    return *this;
}

core::Status CudaQuantizedMatrix::create(
    std::size_t rows,
    std::size_t cols,
    core::WeightFormat format,
    CudaQuantizedMatrix* output) {
    if (output == nullptr) {
        return core::Status::error("CudaQuantizedMatrix output is null");
    }
    std::size_t bytes = 0;
    const core::Status layout_status =
        core::checked_quantized_matrix_bytes(format, rows, cols, &bytes);
    if (!layout_status.is_ok()) return layout_status;

    CudaQuantizedMatrix replacement;
    const CudaStatus allocation = DeviceBuffer::allocate(bytes, &replacement.storage_);
    if (!allocation.is_ok()) return allocation.to_core_status();
    replacement.rows_ = rows;
    replacement.cols_ = cols;
    replacement.format_ = format;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaQuantizedMatrix::copy_from_host_async(
    const void* source,
    std::size_t bytes,
    const CudaStream& stream) {
    if (bytes != size_bytes()) {
        return core::Status::error("CudaQuantizedMatrix H2D byte count mismatch");
    }
    return storage_.copy_from_host_async(source, bytes, stream).to_core_status();
}

core::Status CudaQuantizedMatrix::reset() {
    const CudaStatus status = storage_.reset();
    rows_ = 0;
    cols_ = 0;
    format_ = core::WeightFormat::Unknown;
    return status.to_core_status();
}

const void* CudaQuantizedMatrix::data() const { return storage_.data(); }

std::size_t CudaQuantizedMatrix::rows() const { return rows_; }

std::size_t CudaQuantizedMatrix::cols() const { return cols_; }

std::size_t CudaQuantizedMatrix::size_bytes() const { return storage_.size_bytes(); }

core::WeightFormat CudaQuantizedMatrix::format() const { return format_; }

bool CudaQuantizedMatrix::empty() const { return storage_.empty(); }

}  // namespace cpp_llm_engine::cuda
