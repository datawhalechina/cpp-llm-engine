// Engine：预计算 embedding 到 Chapter06 RequestModelExecutor 的适配器
// 职责：让文本或多模态 embedding 复用现有 Runtime prefill/decode 生命周期。
// 契约：Request prompt 只表达 position 数量；真实 hidden 输入由每请求 Context 持有。
// 验证：embedding_request_executor_test 覆盖同步 Runtime、资源释放和错误传播。
#pragma once

#include "chapter06_generation/model_executor.h"
#include "cpp_llm_engine/core/decoder_contract.h"

#include <cstddef>
#include <memory>
#include <string>

namespace cpp_llm_engine {

class EmbeddingRequestExecutor final
    : public chapter06_generation::RequestModelExecutor {
public:
    static std::shared_ptr<EmbeddingRequestExecutor> create(
        std::unique_ptr<core::DecoderExecutionContext> context,
        core::DecoderEmbeddingSequence embeddings,
        std::string* error = nullptr);

    bool prefill(const chapter06_generation::TokenSequence& prompt_positions,
                 chapter06_generation::Logits* logits,
                 std::string* error) override;
    bool decode(chapter06_generation::TokenId token,
                chapter06_generation::Logits* logits,
                std::string* error) override;
    chapter06_generation::ModelExecutionState execution_state() const override;

    bool prefilled() const;
    std::size_t embedding_storage_bytes() const;
    std::size_t context_token_count() const;

private:
    EmbeddingRequestExecutor(
        std::unique_ptr<core::DecoderExecutionContext> context,
        core::DecoderEmbeddingSequence embeddings);

    std::unique_ptr<core::DecoderExecutionContext> context_;
    core::DecoderEmbeddingSequence embeddings_;
    bool prefilled_ = false;
};

}  // namespace cpp_llm_engine
