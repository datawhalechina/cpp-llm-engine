// Engine CUDA：声明真实 Llama 每请求执行器工厂。
// 职责：共享 CUDA Model，为每个请求创建独立 Context 并适配 Chapter06 Runtime。
// 契约：批量 Decode 复用精确 batch 形状的 workspace；分页与 Radix 复用 Chapter07 块契约。
#pragma once

#include "chapter06_generation/model_executor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace chapter07_kv_cache {
class KvCacheTransaction;
}

namespace cpp_llm_engine {

enum class CudaAttentionMode {
    ThreeStage,
    FusedOnline,
};

enum class CudaQuantizedGemvMode {
    FixedTile4,
    Adaptive,
};

struct RealLlamaCudaPrefixCacheStats {
    std::size_t queries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t matched_tokens = 0;
    std::size_t evicted_blocks = 0;
    std::size_t cached_blocks = 0;
};

class RealLlamaCudaExecutorFactory final
    : public chapter06_generation::ExecutorBatchRunner {
public:
    static std::unique_ptr<RealLlamaCudaExecutorFactory> load(
        const std::string& model_path,
        std::size_t context_capacity,
        std::string* error = nullptr);

    ~RealLlamaCudaExecutorFactory();
    RealLlamaCudaExecutorFactory(RealLlamaCudaExecutorFactory&&) noexcept;
    RealLlamaCudaExecutorFactory& operator=(RealLlamaCudaExecutorFactory&&) noexcept;

    RealLlamaCudaExecutorFactory(const RealLlamaCudaExecutorFactory&) = delete;
    RealLlamaCudaExecutorFactory& operator=(const RealLlamaCudaExecutorFactory&) = delete;

    std::shared_ptr<chapter06_generation::RequestModelExecutor>
    create_executor() const;
    void set_attention_mode(CudaAttentionMode mode);
    CudaAttentionMode attention_mode() const;
    void set_quantized_gemv_mode(CudaQuantizedGemvMode mode);
    CudaQuantizedGemvMode quantized_gemv_mode() const;
    bool configure_paged_kv_cache(
        std::size_t total_blocks,
        std::size_t block_size,
        std::string* error = nullptr);
    std::shared_ptr<chapter07_kv_cache::KvCacheTransaction>
    paged_kv_resource() const;
    bool bind_request(
        const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
        std::size_t request_id,
        std::string* error = nullptr) const override;
    void release_request(std::size_t request_id) const override;
    std::size_t estimate_cached_prefix(
        const chapter06_generation::Request& request) const override;
    bool adopt_cached_prefix(
        const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
        std::size_t request_id,
        chapter06_generation::Request& request,
        std::string* error = nullptr) const override;
    void publish_cached_prefix(
        std::size_t request_id,
        const chapter06_generation::Request& request) const override;
    bool prefix_cache_stats(
        RealLlamaCudaPrefixCacheStats* stats,
        std::string* error = nullptr) const;
    bool decode_batch(
        const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>& executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& expected_positions,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const override;
    std::size_t vocab_size() const;
    std::size_t layer_count() const;
    std::int32_t eos_token_id() const;
    std::string architecture() const;
    std::size_t context_capacity() const;
    std::size_t batch_workspace_count() const;
    std::size_t batch_workspace_creation_count() const;
    std::size_t batch_workspace_use_count(std::size_t batch_size) const;
    std::size_t batch_workspace_device_bytes(std::size_t batch_size) const;
    std::size_t batch_workspace_dynamic_allocation_count(
        std::size_t batch_size) const;

private:
    struct Impl;
    explicit RealLlamaCudaExecutorFactory(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
