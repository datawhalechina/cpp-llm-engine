// Engine：预计算 embedding RequestModelExecutor 实现
// 职责：把 Chapter06 的 position-count prefill 转发到 Core embedding prefill。
// 资源：prefill 成功后释放输入 embedding；Context 保留 position 与分层 K/V。
// 验证：embedding_request_executor_test。
#include "cpp_llm_engine/embedding_request_executor.h"

#include <utility>
#include <vector>

namespace cpp_llm_engine {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

}  // namespace

std::shared_ptr<EmbeddingRequestExecutor> EmbeddingRequestExecutor::create(
    std::unique_ptr<core::DecoderExecutionContext> context,
    core::DecoderEmbeddingSequence embeddings,
    std::string* error) {
    if (context == nullptr) {
        fail(error, "embedding request executor requires a decoder context");
        return {};
    }
    const auto contract = context->model_contract();
    const auto contract_status = contract.validate();
    if (!contract_status.is_ok()) {
        fail(error, contract_status.message());
        return {};
    }
    const auto embedding_status = embeddings.validate(contract.layer.hidden_size);
    if (!embedding_status.is_ok()) {
        fail(error, embedding_status.message());
        return {};
    }
    if (embeddings.token_count > contract.max_context_tokens) {
        fail(error, "embedding request exceeds decoder context length");
        return {};
    }
    if (error != nullptr) error->clear();
    return std::shared_ptr<EmbeddingRequestExecutor>(
        new EmbeddingRequestExecutor(std::move(context), std::move(embeddings)));
}

EmbeddingRequestExecutor::EmbeddingRequestExecutor(
    std::unique_ptr<core::DecoderExecutionContext> context,
    core::DecoderEmbeddingSequence embeddings)
    : context_(std::move(context)), embeddings_(std::move(embeddings)) {}

bool EmbeddingRequestExecutor::prefill(
    const chapter06_generation::TokenSequence& prompt_positions,
    chapter06_generation::Logits* logits,
    std::string* error) {
    if (prefilled_) return fail(error, "embedding request is already prefilled");
    if (logits == nullptr) return fail(error, "embedding prefill logits must not be null");
    if (prompt_positions.size() != embeddings_.token_count) {
        return fail(error, "request position count does not match embedding rows");
    }
    if (context_->token_count() != 0) {
        return fail(error, "embedding prefill requires an empty decoder context");
    }
    const auto status = context_->prefill_embeddings(embeddings_, logits);
    if (!status.is_ok()) return fail(error, status.message());
    if (logits->empty()) return fail(error, "embedding prefill returned empty logits");
    prefilled_ = true;
    std::vector<float>().swap(embeddings_.values);
    if (error != nullptr) error->clear();
    return true;
}

bool EmbeddingRequestExecutor::decode(
    chapter06_generation::TokenId token,
    chapter06_generation::Logits* logits,
    std::string* error) {
    if (!prefilled_) return fail(error, "embedding request must be prefilled before decode");
    if (logits == nullptr) return fail(error, "embedding decode logits must not be null");
    const auto status = context_->decode_token(token, logits);
    if (!status.is_ok()) return fail(error, status.message());
    if (logits->empty()) return fail(error, "embedding decode returned empty logits");
    if (error != nullptr) error->clear();
    return true;
}

chapter06_generation::ModelExecutionState
EmbeddingRequestExecutor::execution_state() const {
    const auto contract = context_->model_contract();
    return {
        true,
        context_->token_count(),
        contract.layer_count,
        contract.layer.uses_kv_cache,
    };
}

bool EmbeddingRequestExecutor::prefilled() const {
    return prefilled_;
}

std::size_t EmbeddingRequestExecutor::embedding_storage_bytes() const {
    return embeddings_.values.capacity() * sizeof(float);
}

std::size_t EmbeddingRequestExecutor::context_token_count() const {
    return context_->token_count();
}

}  // namespace cpp_llm_engine
