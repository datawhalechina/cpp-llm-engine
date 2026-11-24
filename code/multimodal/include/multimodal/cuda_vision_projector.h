// 多模态 CUDA：IDEFICS3 空间合并与 Projector 设备接口
// 职责：保持视觉 hidden states 在设备端完成 pixel-unshuffle 等价重排和 Linear。
// 契约：输入网格为 row-major，合并顺序为 dy、dx、hidden；Projector 无 bias。
// 验证：multimodal_cuda_vision_projector_test 与 CPU reference 对照。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "multimodal/vision_projector.h"

#include <cstddef>

namespace cpp_llm_engine::cuda {
class CudaExecutionContext;
}

namespace multimodal {

class CudaIdefics3Projector final {
public:
    static cpp_llm_engine::core::Status upload(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const LinearWeights& host,
        CudaIdefics3Projector* output);

    cpp_llm_engine::core::Status forward(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const cpp_llm_engine::cuda::CudaTensor& input,
        std::size_t grid_width,
        std::size_t grid_height,
        std::size_t merge_size,
        cpp_llm_engine::cuda::CudaTensor* output) const;
    cpp_llm_engine::core::Status forward_batch(
        cpp_llm_engine::cuda::CudaExecutionContext* context,
        const cpp_llm_engine::cuda::CudaTensor& input,
        std::size_t batch_size,
        std::size_t grid_width,
        std::size_t grid_height,
        std::size_t merge_size,
        cpp_llm_engine::cuda::CudaTensor* output) const;

    bool valid() const;
    std::size_t input_size() const;
    std::size_t output_size() const;

private:
    cpp_llm_engine::cuda::CudaTensor weight_;
    std::size_t input_size_ = 0;
    std::size_t output_size_ = 0;
};

}  // namespace multimodal
