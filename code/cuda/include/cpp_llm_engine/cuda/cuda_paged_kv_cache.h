// CUDA 执行层：声明共享物理块组成的分页 K/V payload。
// 职责：按 layer/block/slot/head/dimension 在设备上保存 K/V，不拥有请求页表。
// 契约：调用方先通过 Chapter07 预留物理块；写入失败不改变控制面 token 计数。
// 验证：真实 GPU 测试使用非连续 BlockTable 写入、读回、清块和块复用。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/device_buffer.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;
class CudaTensor;

class CudaPagedKvCache final {
public:
    CudaPagedKvCache() = default;
    ~CudaPagedKvCache() = default;

    CudaPagedKvCache(const CudaPagedKvCache&) = delete;
    CudaPagedKvCache& operator=(const CudaPagedKvCache&) = delete;
    CudaPagedKvCache(CudaPagedKvCache&&) noexcept = default;
    CudaPagedKvCache& operator=(CudaPagedKvCache&&) noexcept = default;

    static core::Status create(
        std::size_t layer_count,
        std::size_t total_blocks,
        std::size_t block_size,
        std::size_t kv_head_count,
        std::size_t head_dimension,
        CudaPagedKvCache* output);

    core::Status write_token(
        std::size_t layer,
        std::size_t physical_block,
        std::size_t slot,
        const CudaTensor& key,
        const CudaTensor& value,
        const CudaStream& stream);
    core::Status copy_token_to_host_async(
        std::size_t layer,
        std::size_t physical_block,
        std::size_t slot,
        float* key,
        float* value,
        const CudaStream& stream) const;
    core::Status clear_block(
        std::size_t physical_block,
        const CudaStream& stream);
    core::Status reset();

    const float* key_data() const;
    const float* value_data() const;
    std::size_t layer_count() const;
    std::size_t total_blocks() const;
    std::size_t block_size() const;
    std::size_t kv_head_count() const;
    std::size_t head_dimension() const;
    std::size_t token_width() const;
    std::size_t payload_bytes() const;
    bool valid() const;

private:
    bool locate(
        std::size_t layer,
        std::size_t physical_block,
        std::size_t slot,
        std::size_t* byte_offset) const;

    DeviceBuffer key_;
    DeviceBuffer value_;
    std::size_t layer_count_ = 0;
    std::size_t total_blocks_ = 0;
    std::size_t block_size_ = 0;
    std::size_t kv_head_count_ = 0;
    std::size_t head_dimension_ = 0;
    std::size_t one_payload_bytes_ = 0;
};

}  // namespace cpp_llm_engine::cuda
