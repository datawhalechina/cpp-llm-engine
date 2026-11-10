// Engine：声明 Qwen2MoE CPU 每请求执行器工厂。
// 职责：共享只读 Model，为每个请求创建独立的 24 层 Context。
#pragma once

#include "chapter06_generation/model_executor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cpp_llm_engine {

struct RealQwen2MoeLayerCacheStats {
    std::size_t capacity = 0;
    std::size_t resident_layers = 0;
    std::size_t resident_bytes = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
};

enum class RealQwen2MoeBatchMode : std::uint8_t {
    RowWise = 0,
    Grouped,
};

class RealQwen2MoeExecutorFactory final
    : public chapter06_generation::ExecutorBatchRunner {
public:
    static std::unique_ptr<RealQwen2MoeExecutorFactory> load(
        const std::string& model_path,
        std::string* error = nullptr);

    std::shared_ptr<chapter06_generation::RequestModelExecutor>
    create_executor() const;
    std::size_t layer_count() const;
    std::size_t vocab_size() const;
    chapter06_generation::TokenId eos_token_id() const;
    bool decode_batch(
        const std::vector<std::shared_ptr<
            chapter06_generation::RequestModelExecutor>>& executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& expected_positions,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const override;
    void set_layer_cache_capacity(std::size_t layer_count);
    RealQwen2MoeLayerCacheStats layer_cache_stats() const;
    void set_moe_batch_mode(RealQwen2MoeBatchMode mode);
    RealQwen2MoeBatchMode moe_batch_mode() const;

private:
    struct Impl;
    explicit RealQwen2MoeExecutorFactory(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
