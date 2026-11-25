// Engine CUDA：预计算 embedding 请求的 Prefill/Decode 组批适配器。
// 职责：统一校验每请求 CUDA Context，并为 Decode 复用精确 batch 形状的工作区。
// 契约：Prefill 原子提交，位置和 K/V 历史仍由每请求 Context 持有。
#pragma once

#include "chapter06_generation/model_executor.h"

#include <cstddef>
#include <memory>

namespace cpp_llm_engine::model {
class RealLlamaCudaModel;
}

namespace cpp_llm_engine {

enum class CudaEmbeddingPrefillPath {
    Adaptive,
    SerialContexts,
    PackedEqualLength,
};

class CudaEmbeddingBatchRunner final
    : public chapter06_generation::ExecutorBatchRunner {
public:
    explicit CudaEmbeddingBatchRunner(
        const model::RealLlamaCudaModel& model,
        CudaEmbeddingPrefillPath prefill_path =
            CudaEmbeddingPrefillPath::Adaptive);
    ~CudaEmbeddingBatchRunner();

    CudaEmbeddingBatchRunner(CudaEmbeddingBatchRunner&&) noexcept;
    CudaEmbeddingBatchRunner& operator=(CudaEmbeddingBatchRunner&&) noexcept;

    CudaEmbeddingBatchRunner(const CudaEmbeddingBatchRunner&) = delete;
    CudaEmbeddingBatchRunner& operator=(const CudaEmbeddingBatchRunner&) = delete;

    bool supports_prefill_batch() const override;
    bool prefill_batch(
        const std::vector<std::shared_ptr<
            chapter06_generation::RequestModelExecutor>>& executors,
        const std::vector<chapter06_generation::TokenSequence>& prompts,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const override;

    bool decode_batch(
        const std::vector<std::shared_ptr<
            chapter06_generation::RequestModelExecutor>>& executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& expected_positions,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const override;

    std::size_t workspace_count() const;
    std::size_t prefill_workspace_count() const;
    std::size_t packed_prefill_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
