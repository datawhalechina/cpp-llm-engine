// Engine CUDA：预计算 embedding 请求的批量 Decode 适配器。
// 职责：提取每请求 CUDA Context，校验位置并复用精确 batch 形状的工作区。
// 契约：只合并 Decode 计算；Prefill、位置和 K/V 历史仍由每请求 Context 持有。
#pragma once

#include "chapter06_generation/model_executor.h"

#include <cstddef>
#include <memory>

namespace cpp_llm_engine::model {
class RealLlamaCudaModel;
}

namespace cpp_llm_engine {

class CudaEmbeddingBatchRunner final
    : public chapter06_generation::ExecutorBatchRunner {
public:
    explicit CudaEmbeddingBatchRunner(const model::RealLlamaCudaModel& model);
    ~CudaEmbeddingBatchRunner();

    CudaEmbeddingBatchRunner(CudaEmbeddingBatchRunner&&) noexcept;
    CudaEmbeddingBatchRunner& operator=(CudaEmbeddingBatchRunner&&) noexcept;

    CudaEmbeddingBatchRunner(const CudaEmbeddingBatchRunner&) = delete;
    CudaEmbeddingBatchRunner& operator=(const CudaEmbeddingBatchRunner&) = delete;

    bool decode_batch(
        const std::vector<std::shared_ptr<
            chapter06_generation::RequestModelExecutor>>& executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& expected_positions,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const override;

    std::size_t workspace_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
