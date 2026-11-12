// Engine CUDA：把 DeepSeek2 CUDA Model/Context 适配到 Chapter06 Runtime。
// 所有权：Factory 共享模型权重，每个 Executor 独占一个 MLA Context。
#pragma once

#include "chapter06_generation/model_executor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cpp_llm_engine {

class RealDeepSeek2CudaExecutorFactory final
    : public chapter06_generation::ExecutorBatchRunner {
public:
    static std::unique_ptr<RealDeepSeek2CudaExecutorFactory> load(
        const std::string& model_path,
        std::size_t context_capacity,
        std::string* error = nullptr);

    ~RealDeepSeek2CudaExecutorFactory();
    RealDeepSeek2CudaExecutorFactory(
        RealDeepSeek2CudaExecutorFactory&&) noexcept;
    RealDeepSeek2CudaExecutorFactory& operator=(
        RealDeepSeek2CudaExecutorFactory&&) noexcept;

    RealDeepSeek2CudaExecutorFactory(
        const RealDeepSeek2CudaExecutorFactory&) = delete;
    RealDeepSeek2CudaExecutorFactory& operator=(
        const RealDeepSeek2CudaExecutorFactory&) = delete;

    std::shared_ptr<chapter06_generation::RequestModelExecutor>
    create_executor() const;
    bool decode_batch(
        const std::vector<std::shared_ptr<
            chapter06_generation::RequestModelExecutor>>& executors,
        const chapter06_generation::TokenSequence& tokens,
        const std::vector<std::size_t>& expected_positions,
        const std::vector<std::size_t>& request_ids,
        std::vector<chapter06_generation::Logits>* logits,
        std::string* error = nullptr) const override;
    std::size_t layer_count() const;
    std::size_t vocab_size() const;
    std::int32_t eos_token_id() const;
    std::size_t context_capacity() const;
    std::size_t batch_workspace_count() const;
    std::size_t batch_workspace_use_count(std::size_t batch_size) const;
    std::size_t grouped_moe_call_count(std::size_t batch_size) const;
    std::size_t active_expert_group_count(std::size_t batch_size) const;
    std::size_t multi_row_expert_group_count(std::size_t batch_size) const;
    std::size_t routed_row_count(std::size_t batch_size) const;

private:
    struct Impl;
    explicit RealDeepSeek2CudaExecutorFactory(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_llm_engine
