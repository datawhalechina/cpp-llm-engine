// CUDA 执行层：声明拥有 packed 量化 payload 的设备矩阵。
// 职责：绑定 row-major rows/cols、量化格式和 device storage。
// 验证：Quantized Linear CTest 覆盖上传、移动、格式和字节数错误。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"
#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/device_buffer.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;

class CudaQuantizedMatrix final {
public:
    CudaQuantizedMatrix() = default;
    ~CudaQuantizedMatrix() = default;

    CudaQuantizedMatrix(const CudaQuantizedMatrix&) = delete;
    CudaQuantizedMatrix& operator=(const CudaQuantizedMatrix&) = delete;
    CudaQuantizedMatrix(CudaQuantizedMatrix&& other) noexcept;
    CudaQuantizedMatrix& operator=(CudaQuantizedMatrix&& other) noexcept;

    static core::Status create(
        std::size_t rows,
        std::size_t cols,
        core::WeightFormat format,
        CudaQuantizedMatrix* output);

    core::Status copy_from_host_async(
        const void* source,
        std::size_t bytes,
        const CudaStream& stream);
    core::Status reset();

    const void* data() const;
    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t size_bytes() const;
    core::WeightFormat format() const;
    bool empty() const;

private:
    DeviceBuffer storage_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    core::WeightFormat format_ = core::WeightFormat::Unknown;
};

}  // namespace cpp_llm_engine::cuda
