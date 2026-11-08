// CUDA 执行层：声明请求级连续 device KV Cache。
// 职责：独占 [capacity, kv_heads, head_dim] K/V storage，并按 token 追加。
// 契约：append 使用同一显式 stream 执行 D2D；clear 保留容量，reset 释放显存。
// 验证：真实 GPU 测试覆盖偏移、容量、shape、移动、复用和请求隔离。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/device_buffer.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;
class CudaTensor;

class CudaKvCache final {
public:
    CudaKvCache() = default;
    ~CudaKvCache() = default;

    CudaKvCache(const CudaKvCache&) = delete;
    CudaKvCache& operator=(const CudaKvCache&) = delete;
    CudaKvCache(CudaKvCache&&) noexcept = default;
    CudaKvCache& operator=(CudaKvCache&&) noexcept = default;

    static core::Status create(
        std::size_t capacity_tokens,
        std::size_t kv_head_count,
        std::size_t head_dimension,
        CudaKvCache* output);

    core::Status append(
        const CudaTensor& key,
        const CudaTensor& value,
        const CudaStream& stream);
    core::Status copy_to_host_async(
        float* key,
        float* value,
        std::size_t token_count,
        const CudaStream& stream) const;
    core::Status trim(std::size_t token_count);
    void clear();
    core::Status reset();

    const float* key_data() const;
    const float* value_data() const;
    std::size_t size_tokens() const;
    std::size_t capacity_tokens() const;
    std::size_t kv_head_count() const;
    std::size_t head_dimension() const;
    std::size_t token_width() const;
    bool valid() const;

private:
    DeviceBuffer key_;
    DeviceBuffer value_;
    std::size_t size_tokens_ = 0;
    std::size_t capacity_tokens_ = 0;
    std::size_t kv_head_count_ = 0;
    std::size_t head_dimension_ = 0;
};

}  // namespace cpp_llm_engine::cuda
