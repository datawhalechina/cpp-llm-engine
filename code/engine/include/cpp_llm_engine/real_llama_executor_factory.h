// Engine：真实 Llama/Qwen3 CPU 每请求执行器工厂。
// 职责：共享一份只读 CPU Model，并为每个请求创建独立 Context/Executor。
// 验证：真实服务 CTest 检查双请求 golden token、状态隔离和终态释放。
#pragma once

#include "chapter06_generation/model_executor.h"
#include "cpp_llm_engine/core/decoder_contract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace chapter07_kv_cache {
class CpuPagedKvCache;
class KvCacheTransaction;
}

namespace cpp_llm_engine {

enum class RealLlamaExecutionMode : std::uint8_t {
    Scalar,
    OptimizedAvx2,
};

struct RealLlamaExecutorProfile {
    std::size_t forward_tokens = 0;
    double total_ms = 0.0;
    double embedding_ms = 0.0;
    double normalization_ms = 0.0;
    double qkv_projection_ms = 0.0;
    double rope_ms = 0.0;
    double attention_ms = 0.0;
    double attention_output_ms = 0.0;
    double ffn_ms = 0.0;
    double ffn_gate_ms = 0.0;
    double ffn_up_ms = 0.0;
    double ffn_activation_ms = 0.0;
    double ffn_down_ms = 0.0;
    double ffn_residual_ms = 0.0;
    double lm_head_ms = 0.0;
    std::size_t batch_linear_calls = 0;
    double batch_input_pack_ms = 0.0;
    double batch_kernel_ms = 0.0;
    double batch_output_unpack_ms = 0.0;
};

struct RealLlamaExecutorMemory {
    std::size_t committed_tokens = 0;
    std::size_t kv_cache_bytes = 0;
};

struct RealLlamaPrefixCacheStats {
    std::size_t queries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t matched_tokens = 0;
    std::size_t evicted_blocks = 0;
    std::size_t cached_blocks = 0;
};

class RealLlamaExecutorFactory final : public chapter06_generation::ExecutorBatchRunner {
public:
    static std::unique_ptr<RealLlamaExecutorFactory> load(
        const std::string& model_path,
        std::string* error = nullptr);
    static std::unique_ptr<RealLlamaExecutorFactory> load(
        const std::string& model_path,
        RealLlamaExecutionMode mode,
        std::string* error = nullptr);

    ~RealLlamaExecutorFactory();
    RealLlamaExecutorFactory(RealLlamaExecutorFactory&&) noexcept;
    RealLlamaExecutorFactory& operator=(RealLlamaExecutorFactory&&) noexcept;

    RealLlamaExecutorFactory(const RealLlamaExecutorFactory&) = delete;
    RealLlamaExecutorFactory& operator=(const RealLlamaExecutorFactory&) = delete;

    std::shared_ptr<chapter06_generation::RequestModelExecutor> create_executor() const;
    bool configure_paged_kv_cache(
        std::size_t total_blocks,
        std::size_t block_size,
        std::string* error = nullptr);
    std::shared_ptr<chapter07_kv_cache::KvCacheTransaction> paged_kv_resource() const;
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
    bool decode_batch(
        const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>& executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& expected_positions,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const override;
    bool decode_batch(
        const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>& executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& expected_positions,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const;
    bool sync_paged_kv(
        const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
        std::size_t request_id,
        chapter07_kv_cache::CpuPagedKvCache* cache,
        std::string* error = nullptr) const;
    bool decode_batch_paged(
        const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>& executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& expected_positions,
        const std::vector<std::size_t>& request_ids,
        chapter07_kv_cache::CpuPagedKvCache* cache,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const;
    bool set_executor_profiling(
        const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
        bool enabled,
        std::string* error = nullptr) const;
    bool executor_profile(
        const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
        RealLlamaExecutorProfile* profile,
        std::string* error = nullptr) const;
    bool executor_memory(
        const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
        RealLlamaExecutorMemory* memory,
        std::string* error = nullptr) const;
    bool prefix_cache_stats(
        RealLlamaPrefixCacheStats* stats,
        std::string* error = nullptr) const;
    bool executor_uses_paged_kv(
        const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor) const;
    std::size_t model_weight_payload_bytes() const;
    std::string architecture() const;
    std::size_t vocab_size() const;
    chapter06_generation::TokenId eos_token_id() const;
    std::size_t layer_count() const;
    cpp_llm_engine::core::DecoderModelContract decoder_contract() const;
    RealLlamaExecutionMode execution_mode() const;

private:
    struct Impl;
    explicit RealLlamaExecutorFactory(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
