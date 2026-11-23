// DeepSeek2 CUDA Model：管理共享设备权重和请求级 MLA 状态。
#pragma once

#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/cuda/mla_attention_backend.h"
#include "cpp_llm_engine/cuda/quantized_gemv_backend.h"
#include "cpp_llm_engine/model/cuda_deepseek2_moe_feed_forward.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cpp_llm_engine::cuda {
class CudaPagedMlaCache;
}

namespace cpp_llm_engine::model {

class RealDeepSeek2CudaContext;
class RealDeepSeek2CudaBatchWorkspace;

class RealDeepSeek2CudaMlaDataPlane {
public:
    virtual ~RealDeepSeek2CudaMlaDataPlane() = default;
    virtual std::size_t request_id() const = 0;
    virtual std::size_t token_count() const = 0;
    virtual bool reserve(std::size_t total_tokens, std::string* error) = 0;
    virtual bool commit(std::size_t total_tokens, std::string* error) = 0;
    virtual bool trim(std::size_t total_tokens) = 0;
    virtual std::vector<std::size_t> physical_blocks() const = 0;
    virtual cuda::CudaPagedMlaCache* payload() const = 0;
};

struct RealDeepSeek2CudaBatchItem final {
    RealDeepSeek2CudaContext* context = nullptr;
    std::int32_t token = 0;
};

class RealDeepSeek2CudaModel final {
public:
    static std::unique_ptr<RealDeepSeek2CudaModel> load(
        const std::string& model_path,
        std::size_t layer_count = 0,
        std::string* error = nullptr);

    ~RealDeepSeek2CudaModel();
    RealDeepSeek2CudaModel(RealDeepSeek2CudaModel&&) noexcept;
    RealDeepSeek2CudaModel& operator=(RealDeepSeek2CudaModel&&) noexcept;
    RealDeepSeek2CudaModel(const RealDeepSeek2CudaModel&) = delete;
    RealDeepSeek2CudaModel& operator=(const RealDeepSeek2CudaModel&) = delete;

    std::unique_ptr<RealDeepSeek2CudaContext> create_context(
        std::size_t capacity_tokens,
        std::string* error = nullptr,
        cuda::CudaMlaComputePath mla_path =
            cuda::CudaMlaComputePath::LegacyThreeStage,
        CudaMoePrefillPath moe_prefill_path =
            CudaMoePrefillPath::DeviceRouterPackedDispatch,
        cuda::CudaQuantizedGemvPath gemv_path =
            cuda::CudaQuantizedGemvPath::Adaptive) const;
    std::unique_ptr<RealDeepSeek2CudaBatchWorkspace> create_batch_workspace(
        std::size_t batch_size,
        std::string* error = nullptr,
        cuda::CudaQuantizedGemvPath gemv_path =
            cuda::CudaQuantizedGemvPath::Adaptive) const;
    bool decode_batch(
        const std::vector<RealDeepSeek2CudaBatchItem>& items,
        RealDeepSeek2CudaBatchWorkspace* workspace,
        std::vector<std::vector<float>>* logits,
        std::string* error = nullptr) const;
    std::size_t layer_count() const;
    std::size_t hidden_size() const;
    std::size_t mla_latent_dimension() const;
    std::size_t mla_rope_dimension() const;
    std::size_t vocab_size() const;
    std::size_t max_context_tokens() const;
    std::int32_t eos_token_id() const;

private:
    friend class RealDeepSeek2CudaContext;
    friend class RealDeepSeek2CudaBatchWorkspace;
    struct Impl;
    explicit RealDeepSeek2CudaModel(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
};

class RealDeepSeek2CudaContext final {
public:
    ~RealDeepSeek2CudaContext();
    RealDeepSeek2CudaContext(RealDeepSeek2CudaContext&&) noexcept;
    RealDeepSeek2CudaContext& operator=(RealDeepSeek2CudaContext&&) noexcept;
    RealDeepSeek2CudaContext(const RealDeepSeek2CudaContext&) = delete;
    RealDeepSeek2CudaContext& operator=(const RealDeepSeek2CudaContext&) = delete;

    core::Status forward_hidden(
        const std::vector<float>& input,
        std::vector<float>* output);
    core::Status forward_token(
        std::int32_t token,
        std::vector<float>* logits);
    core::Status forward_tokens(
        const std::vector<std::int32_t>& tokens,
        std::vector<float>* logits);
    void reset();
    bool attach_mla_data_plane(
        std::shared_ptr<RealDeepSeek2CudaMlaDataPlane> data_plane,
        std::string* error = nullptr);
    bool uses_external_mla() const;
    bool uses_external_mla_for(std::size_t request_id) const;
    std::size_t position() const;
    std::size_t layer_count() const;

private:
    friend class RealDeepSeek2CudaModel;
    struct Impl;
    explicit RealDeepSeek2CudaContext(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class RealDeepSeek2CudaBatchWorkspace final {
public:
    ~RealDeepSeek2CudaBatchWorkspace();
    RealDeepSeek2CudaBatchWorkspace(
        RealDeepSeek2CudaBatchWorkspace&&) noexcept;
    RealDeepSeek2CudaBatchWorkspace& operator=(
        RealDeepSeek2CudaBatchWorkspace&&) noexcept;
    RealDeepSeek2CudaBatchWorkspace(
        const RealDeepSeek2CudaBatchWorkspace&) = delete;
    RealDeepSeek2CudaBatchWorkspace& operator=(
        const RealDeepSeek2CudaBatchWorkspace&) = delete;

    std::size_t batch_size() const;
    std::size_t use_count() const;
    std::size_t grouped_moe_calls() const;
    std::size_t active_expert_groups() const;
    std::size_t multi_row_expert_groups() const;
    std::size_t routed_rows() const;

private:
    friend class RealDeepSeek2CudaModel;
    struct Impl;
    explicit RealDeepSeek2CudaBatchWorkspace(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine::model
