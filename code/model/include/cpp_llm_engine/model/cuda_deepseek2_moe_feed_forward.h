// DeepSeek2 CUDA Model：声明 Host 路由、设备执行的 MoE FFN。
// 职责：在 CUDA 上执行命中的量化专家和共享专家。
// 契约：全部专家成功后才写入调用方输出。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/cuda_linear_weight.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace chapter01 { struct RealDeepSeek2MoeLayerWeights; }
namespace cpp_llm_engine::cuda { class CudaExecutionContext; }

namespace cpp_llm_engine::model {

struct CudaDeepSeek2MoeConfig final {
    std::size_t hidden_size = 0;
    std::size_t expert_hidden_size = 0;
    std::size_t shared_expert_hidden_size = 0;
    std::size_t expert_count = 0;
    std::size_t top_k = 0;
    bool normalize_weights = false;
    float weight_scale = 1.0F;

    core::Status validate() const;
};

struct CudaDeepSeek2MoeRoute final {
    std::vector<std::size_t> expert_indices;
    std::vector<float> weights;
};

class CudaDeepSeek2MoeWeights final {
public:
    static core::Status upload(
        cuda::CudaExecutionContext* context,
        const chapter01::RealDeepSeek2MoeLayerWeights& host,
        CudaDeepSeek2MoeWeights* output);

    bool valid() const;
    std::size_t hidden_size() const;
    std::size_t expert_hidden_size() const;
    std::size_t shared_expert_hidden_size() const;
    std::size_t expert_count() const;
    const std::vector<float>& router() const;
    const cuda::CudaQuantizedMatrix& expert_gate(std::size_t index) const;
    const cuda::CudaQuantizedMatrix& expert_up(std::size_t index) const;
    const cuda::CudaQuantizedMatrix& expert_down(std::size_t index) const;
    const cuda::CudaQuantizedMatrix& shared_gate() const;
    const cuda::CudaQuantizedMatrix& shared_up() const;
    const cuda::CudaQuantizedMatrix& shared_down() const;
    const cuda::CudaTensor& attention_norm() const;
    cuda::CudaLinearWeight query() const;
    cuda::CudaLinearWeight kv_a_mqa() const;
    const cuda::CudaTensor& kv_a_norm() const;
    const cuda::CudaTensor& kv_b_dequantized() const;
    cuda::CudaLinearWeight attention_output() const;
    const cuda::CudaTensor& ffn_norm() const;

private:
    std::vector<float> router_;
    std::vector<cuda::CudaQuantizedMatrix> expert_gate_;
    std::vector<cuda::CudaQuantizedMatrix> expert_up_;
    std::vector<cuda::CudaQuantizedMatrix> expert_down_;
    cuda::CudaQuantizedMatrix shared_gate_;
    cuda::CudaQuantizedMatrix shared_up_;
    cuda::CudaQuantizedMatrix shared_down_;
    cuda::CudaTensor attention_norm_;
    cuda::CudaQuantizedMatrix query_;
    cuda::CudaQuantizedMatrix kv_a_mqa_;
    cuda::CudaTensor kv_a_norm_;
    cuda::CudaQuantizedMatrix kv_b_;
    cuda::CudaTensor kv_b_dequantized_;
    cuda::CudaQuantizedMatrix attention_output_;
    cuda::CudaTensor ffn_norm_;
    std::size_t hidden_size_ = 0;
    std::size_t expert_hidden_size_ = 0;
    std::size_t shared_expert_hidden_size_ = 0;
};

class CudaDeepSeek2MoeFeedForward final {
public:
    static core::Status create(
        const CudaDeepSeek2MoeConfig& config,
        CudaDeepSeek2MoeFeedForward* output);

    core::Status forward(
        cuda::CudaExecutionContext* context,
        const CudaDeepSeek2MoeWeights& weights,
        const cuda::CudaTensor& normalized_hidden,
        cuda::CudaTensor* output,
        CudaDeepSeek2MoeRoute* route);

private:
    CudaDeepSeek2MoeConfig config_;
    cuda::CudaTensor gate_;
    cuda::CudaTensor up_;
    cuda::CudaTensor activated_;
    cuda::CudaTensor expert_output_;
    cuda::CudaTensor shared_gate_;
    cuda::CudaTensor shared_up_;
    cuda::CudaTensor shared_activated_;
};

class CudaDeepSeek2MoeBatchWorkspace final {
public:
    static core::Status create(
        const CudaDeepSeek2MoeConfig& config,
        std::size_t batch_size,
        CudaDeepSeek2MoeBatchWorkspace* output);

    CudaDeepSeek2MoeBatchWorkspace();
    ~CudaDeepSeek2MoeBatchWorkspace();
    CudaDeepSeek2MoeBatchWorkspace(
        CudaDeepSeek2MoeBatchWorkspace&&) noexcept;
    CudaDeepSeek2MoeBatchWorkspace& operator=(
        CudaDeepSeek2MoeBatchWorkspace&&) noexcept;
    CudaDeepSeek2MoeBatchWorkspace(
        const CudaDeepSeek2MoeBatchWorkspace&) = delete;
    CudaDeepSeek2MoeBatchWorkspace& operator=(
        const CudaDeepSeek2MoeBatchWorkspace&) = delete;

    core::Status forward(
        cuda::CudaExecutionContext* context,
        const CudaDeepSeek2MoeWeights& weights,
        const std::vector<const cuda::CudaTensor*>& normalized_hidden,
        const std::vector<cuda::CudaTensor*>& output,
        std::vector<CudaDeepSeek2MoeRoute>* routes);
    std::size_t batch_size() const;
    std::size_t use_count() const;
    std::size_t active_expert_groups() const;
    std::size_t multi_row_expert_groups() const;
    std::size_t routed_rows() const;

private:
    struct Impl;
    explicit CudaDeepSeek2MoeBatchWorkspace(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine::model
