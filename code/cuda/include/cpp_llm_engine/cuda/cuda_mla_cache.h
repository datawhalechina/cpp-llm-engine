// CUDA 执行层：声明请求级压缩 MLA Cache。
// 职责：保存每个词元的 latent K/V 与共享旋转键，避免展开完整多头 K/V。
// 契约：append 在同一 stream 依次写入两段 payload，全部成功入队后推进逻辑词元数。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/device_buffer.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {

class CudaStream;
class CudaTensor;

class CudaMlaCache final {
public:
    static core::Status create(
        std::size_t capacity_tokens,
        std::size_t latent_dimension,
        std::size_t rope_dimension,
        CudaMlaCache* output);

    core::Status append(
        const CudaTensor& latent,
        const CudaTensor& key_rope,
        const CudaStream& stream);
    core::Status copy_to_host_async(
        float* latent,
        float* key_rope,
        std::size_t token_count,
        const CudaStream& stream) const;
    core::Status trim(std::size_t token_count);
    void clear();

    const float* latent_data() const;
    const float* key_rope_data() const;
    std::size_t size_tokens() const;
    std::size_t capacity_tokens() const;
    std::size_t latent_dimension() const;
    std::size_t rope_dimension() const;
    bool valid() const;

private:
    DeviceBuffer latent_;
    DeviceBuffer key_rope_;
    std::size_t size_tokens_ = 0;
    std::size_t capacity_tokens_ = 0;
    std::size_t latent_dimension_ = 0;
    std::size_t rope_dimension_ = 0;
};

}  // namespace cpp_llm_engine::cuda
