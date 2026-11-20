// CUDA 执行层：声明共享物理块组成的分页 MLA payload。
// 职责：按 layer/block/slot 保存 latent K/V 与 RoPE key，不拥有请求页表。
// 契约：调用方先通过 Chapter07 预留物理块，设备写入不推进控制面计数。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/device_buffer.h"

#include <cstddef>
#include <vector>

namespace cpp_llm_engine::cuda {

class CudaStream;
class CudaTensor;

class CudaPagedMlaCache final {
public:
    CudaPagedMlaCache() = default;

    CudaPagedMlaCache(const CudaPagedMlaCache&) = delete;
    CudaPagedMlaCache& operator=(const CudaPagedMlaCache&) = delete;
    CudaPagedMlaCache(CudaPagedMlaCache&&) noexcept = default;
    CudaPagedMlaCache& operator=(CudaPagedMlaCache&&) noexcept = default;

    static core::Status create(
        std::size_t layer_count,
        std::size_t total_blocks,
        std::size_t block_size,
        std::size_t latent_dimension,
        std::size_t rope_dimension,
        CudaPagedMlaCache* output);

    core::Status write_token(
        std::size_t layer,
        std::size_t physical_block,
        std::size_t slot,
        const CudaTensor& latent,
        const CudaTensor& key_rope,
        const CudaStream& stream);
    core::Status write_tokens(
        std::size_t layer,
        const std::vector<std::size_t>& physical_blocks,
        std::size_t start_position,
        const CudaTensor& latent,
        const CudaTensor& key_rope,
        const CudaStream& stream);
    core::Status copy_token_to_host_async(
        std::size_t layer,
        std::size_t physical_block,
        std::size_t slot,
        float* latent,
        float* key_rope,
        const CudaStream& stream) const;
    core::Status clear_block(
        std::size_t physical_block,
        const CudaStream& stream);
    core::Status reset();

    const float* latent_data() const;
    const float* key_rope_data() const;
    std::size_t layer_count() const;
    std::size_t total_blocks() const;
    std::size_t block_size() const;
    std::size_t latent_dimension() const;
    std::size_t rope_dimension() const;
    std::size_t payload_bytes() const;
    bool valid() const;

private:
    bool locate(
        std::size_t layer,
        std::size_t physical_block,
        std::size_t slot,
        std::size_t width,
        std::size_t* byte_offset) const;

    DeviceBuffer latent_;
    DeviceBuffer key_rope_;
    std::size_t layer_count_ = 0;
    std::size_t total_blocks_ = 0;
    std::size_t block_size_ = 0;
    std::size_t latent_dimension_ = 0;
    std::size_t rope_dimension_ = 0;
    std::size_t latent_bytes_ = 0;
    std::size_t rope_bytes_ = 0;
};

}  // namespace cpp_llm_engine::cuda
