// Llama CUDA Model：声明共享模型权重与请求级执行上下文。
// 职责：Model 持有只读 device 权重，Context 持有 stream、中间张量、KV Cache 和位置。
// 契约：单次 forward_hidden 要么提交所有层的 KV，要么回退到调用前状态。
#pragma once

#include "cpp_llm_engine/core/status.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cpp_llm_engine::cuda {
class CudaPagedKvCache;
enum class CudaAttentionPath;
enum class CudaQuantizedGemvPath;
}

namespace cpp_llm_engine::model {

class RealLlamaCudaContext;
class RealLlamaCudaBatchWorkspace;

class RealLlamaCudaKvDataPlane {
public:
    virtual ~RealLlamaCudaKvDataPlane() = default;
    virtual std::size_t request_id() const = 0;
    virtual std::size_t token_count() const = 0;
    virtual bool reserve(std::size_t total_tokens, std::string* error) = 0;
    virtual bool commit(std::size_t total_tokens, std::string* error) = 0;
    virtual bool trim(std::size_t total_tokens) = 0;
    virtual std::vector<std::size_t> physical_blocks() const = 0;
    virtual cuda::CudaPagedKvCache* payload() const = 0;
};

struct RealLlamaCudaBatchItem final {
    RealLlamaCudaContext* context = nullptr;
    std::int32_t token = 0;
};

class RealLlamaCudaModel final {
public:
    static std::unique_ptr<RealLlamaCudaModel> load(
        const std::string& model_path,
        std::size_t layer_count = 0,
        std::string* error = nullptr);

    ~RealLlamaCudaModel();
    RealLlamaCudaModel(RealLlamaCudaModel&&) noexcept;
    RealLlamaCudaModel& operator=(RealLlamaCudaModel&&) noexcept;

    RealLlamaCudaModel(const RealLlamaCudaModel&) = delete;
    RealLlamaCudaModel& operator=(const RealLlamaCudaModel&) = delete;

    std::unique_ptr<RealLlamaCudaContext> create_context(
        std::size_t capacity_tokens,
        std::string* error = nullptr) const;
    std::unique_ptr<RealLlamaCudaBatchWorkspace> create_batch_workspace(
        std::size_t batch_size,
        std::string* error = nullptr) const;
    bool decode_batch(
        const std::vector<RealLlamaCudaBatchItem>& items,
        RealLlamaCudaBatchWorkspace* workspace,
        std::vector<std::vector<float>>* logits,
        std::string* error = nullptr) const;
    std::size_t layer_count() const;
    std::size_t model_layer_count() const;
    std::size_t hidden_size() const;
    std::size_t kv_head_count() const;
    std::size_t head_dimension() const;
    std::size_t vocab_size() const;
    std::size_t max_context_tokens() const;
    std::int32_t eos_token_id() const;

private:
    friend class RealLlamaCudaContext;
    friend class RealLlamaCudaBatchWorkspace;
    struct Impl;
    explicit RealLlamaCudaModel(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
};

class RealLlamaCudaBatchWorkspace final {
public:
    ~RealLlamaCudaBatchWorkspace();
    RealLlamaCudaBatchWorkspace(RealLlamaCudaBatchWorkspace&&) noexcept;
    RealLlamaCudaBatchWorkspace& operator=(RealLlamaCudaBatchWorkspace&&) noexcept;

    RealLlamaCudaBatchWorkspace(const RealLlamaCudaBatchWorkspace&) = delete;
    RealLlamaCudaBatchWorkspace& operator=(const RealLlamaCudaBatchWorkspace&) = delete;

    std::size_t batch_size() const;
    std::size_t use_count() const;
    std::size_t device_bytes() const;
    std::size_t dynamic_allocation_count() const;
    void set_quantized_gemv_path(cuda::CudaQuantizedGemvPath path);

private:
    friend class RealLlamaCudaModel;
    struct Impl;
    explicit RealLlamaCudaBatchWorkspace(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class RealLlamaCudaContext final {
public:
    ~RealLlamaCudaContext();
    RealLlamaCudaContext(RealLlamaCudaContext&&) noexcept;
    RealLlamaCudaContext& operator=(RealLlamaCudaContext&&) noexcept;

    RealLlamaCudaContext(const RealLlamaCudaContext&) = delete;
    RealLlamaCudaContext& operator=(const RealLlamaCudaContext&) = delete;

    core::Status forward_hidden(
        const std::vector<float>& input,
        std::vector<float>* output);
    core::Status forward_hidden_sequence(
        const std::vector<float>& input,
        std::size_t token_count,
        std::vector<float>* output);
    core::Status forward_token(
        std::int32_t token,
        std::vector<float>* logits);
    core::Status prefill_tokens(
        const std::vector<std::int32_t>& tokens,
        std::vector<float>* logits);
    core::Status copy_layer_kv(
        std::size_t layer,
        std::vector<float>* key,
        std::vector<float>* value);
    void set_attention_path(cuda::CudaAttentionPath path);
    void set_quantized_gemv_path(cuda::CudaQuantizedGemvPath path);
    void reset();
    bool attach_kv_data_plane(
        std::shared_ptr<RealLlamaCudaKvDataPlane> data_plane,
        std::string* error = nullptr);
    bool adopt_external_kv_prefix(
        std::size_t token_count,
        std::string* error = nullptr);
    bool uses_external_kv() const;
    bool uses_external_kv_for(std::size_t request_id) const;
    std::size_t position() const;
    std::size_t capacity_tokens() const;
    std::size_t prefill_workspace_creation_count() const;

private:
    friend class RealLlamaCudaModel;
    struct Impl;
    explicit RealLlamaCudaContext(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine::model
