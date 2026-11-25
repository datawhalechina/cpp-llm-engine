// Engine CUDA：实现 embedding 请求的真实 Llama 批量 Decode。
#include "cpp_llm_engine/cuda_embedding_batch_runner.h"

#include "cpp_llm_engine/embedding_request_executor.h"
#include "cpp_llm_engine/model/real_llama_cuda_model.h"

#include <map>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace cpp_llm_engine {

struct CudaEmbeddingBatchRunner::Impl final {
    Impl(
        const model::RealLlamaCudaModel& source,
        CudaEmbeddingPrefillPath path)
        : model(source), prefill_path(path) {}

    const model::RealLlamaCudaModel& model;
    CudaEmbeddingPrefillPath prefill_path =
        CudaEmbeddingPrefillPath::Adaptive;
    mutable std::mutex mutex;
    mutable std::map<std::size_t, std::unique_ptr<
        model::RealLlamaCudaBatchWorkspace>> workspaces;
    mutable std::map<std::pair<std::size_t, std::size_t>, std::unique_ptr<
        model::RealLlamaCudaBatchWorkspace>> prefill_workspaces;
    mutable std::size_t packed_prefill_calls = 0;
};

CudaEmbeddingBatchRunner::CudaEmbeddingBatchRunner(
    const model::RealLlamaCudaModel& model,
    CudaEmbeddingPrefillPath prefill_path)
    : impl_(std::make_unique<Impl>(model, prefill_path)) {}

CudaEmbeddingBatchRunner::~CudaEmbeddingBatchRunner() = default;
CudaEmbeddingBatchRunner::CudaEmbeddingBatchRunner(
    CudaEmbeddingBatchRunner&&) noexcept = default;
CudaEmbeddingBatchRunner& CudaEmbeddingBatchRunner::operator=(
    CudaEmbeddingBatchRunner&&) noexcept = default;

bool CudaEmbeddingBatchRunner::supports_prefill_batch() const { return true; }

bool CudaEmbeddingBatchRunner::prefill_batch(
    const std::vector<std::shared_ptr<
        chapter06_generation::RequestModelExecutor>>& executors,
    const std::vector<chapter06_generation::TokenSequence>& prompts,
    const std::vector<std::size_t>& request_ids,
    std::vector<chapter06_generation::Logits>* logits,
    std::string* error) const {
    if (impl_ == nullptr || executors.empty() ||
        executors.size() != prompts.size() ||
        executors.size() != request_ids.size() || logits == nullptr) {
        if (error != nullptr) *error = "CUDA embedding prefill batch sizes do not match";
        return false;
    }
    std::unordered_set<EmbeddingRequestExecutor*> unique;
    std::unordered_set<std::size_t> unique_request_ids;
    std::vector<EmbeddingRequestExecutor*> prepared;
    prepared.reserve(executors.size());
    std::vector<chapter06_generation::Logits> outputs(executors.size());
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::size_t tokens_per_request = prompts.front().size();
    bool equal_length = tokens_per_request != 0;
    for (const auto& prompt : prompts) {
        equal_length = equal_length && prompt.size() == tokens_per_request;
    }
    const bool use_packed = equal_length &&
        impl_->prefill_path == CudaEmbeddingPrefillPath::PackedEqualLength;
    for (std::size_t index = 0; index < executors.size(); ++index) {
        auto* executor = executors[index] == nullptr ? nullptr
            : dynamic_cast<EmbeddingRequestExecutor*>(
                executors[index]->underlying_executor());
        auto* context = executor == nullptr ? nullptr
            : dynamic_cast<model::RealLlamaCudaContext*>(
                executor->decoder_context());
        if (executor == nullptr || context == nullptr ||
            !unique.insert(executor).second ||
            !unique_request_ids.insert(request_ids[index]).second) {
            for (auto* completed : prepared) completed->rollback_batched_prefill();
            return false;
        }
        const bool ready = use_packed
            ? executor->prepare_packed_prefill(prompts[index], error)
            : executor->prepare_batched_prefill(
                  prompts[index], &outputs[index], error);
        if (!ready) {
            for (auto* completed : prepared) completed->rollback_batched_prefill();
            return false;
        }
        prepared.push_back(executor);
    }
    if (use_packed) {
        const auto key = std::make_pair(executors.size(), tokens_per_request);
        auto found = impl_->prefill_workspaces.find(key);
        if (found == impl_->prefill_workspaces.end()) {
            auto workspace = impl_->model.create_batch_workspace(
                executors.size() * tokens_per_request, error);
            if (workspace == nullptr) {
                for (auto* completed : prepared) {
                    completed->rollback_batched_prefill();
                }
                return false;
            }
            found = impl_->prefill_workspaces.emplace(
                key, std::move(workspace)).first;
        }
        std::vector<model::RealLlamaCudaPrefillItem> items;
        items.reserve(prepared.size());
        for (auto* executor : prepared) {
            items.push_back({
                dynamic_cast<model::RealLlamaCudaContext*>(
                    executor->decoder_context()),
                &executor->embeddings(),
            });
        }
        if (!impl_->model.prefill_embeddings_batch(
                items, found->second.get(), &outputs, error)) {
            for (auto* completed : prepared) {
                completed->rollback_batched_prefill();
            }
            return false;
        }
        ++impl_->packed_prefill_calls;
    }
    for (auto* executor : prepared) executor->commit_batched_prefill();
    *logits = std::move(outputs);
    if (error != nullptr) error->clear();
    return true;
}

bool CudaEmbeddingBatchRunner::decode_batch(
    const std::vector<std::shared_ptr<
        chapter06_generation::RequestModelExecutor>>& executors,
    const chapter06_generation::TokenSequence& tokens,
    const std::vector<std::size_t>& expected_positions,
    const std::vector<std::size_t>& request_ids,
    std::vector<chapter06_generation::Logits>* logits,
    std::string* error) const {
    if (impl_ == nullptr || executors.empty() ||
        executors.size() != tokens.size() ||
        executors.size() != expected_positions.size() ||
        executors.size() != request_ids.size() || logits == nullptr) {
        if (error != nullptr) {
            *error = "CUDA embedding batch input sizes do not match";
        }
        return false;
    }

    std::unordered_set<EmbeddingRequestExecutor*> unique;
    std::unordered_set<std::size_t> unique_request_ids;
    std::vector<model::RealLlamaCudaBatchItem> items;
    items.reserve(executors.size());
    for (std::size_t index = 0; index < executors.size(); ++index) {
        auto* executor = executors[index] == nullptr ? nullptr
            : dynamic_cast<EmbeddingRequestExecutor*>(
                executors[index]->underlying_executor());
        auto* context = executor == nullptr ? nullptr
            : dynamic_cast<model::RealLlamaCudaContext*>(
                executor->decoder_context());
        if (executor == nullptr || context == nullptr ||
            !unique.insert(executor).second ||
            !unique_request_ids.insert(request_ids[index]).second ||
            !executor->prefilled() ||
            executor->context_token_count() != expected_positions[index]) {
            if (error != nullptr) {
                *error = "CUDA embedding batch requires unique aligned contexts";
            }
            return false;
        }
        items.push_back({context, tokens[index]});
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto found = impl_->workspaces.find(items.size());
    if (found == impl_->workspaces.end()) {
        auto workspace = impl_->model.create_batch_workspace(items.size(), error);
        if (workspace == nullptr) return false;
        found = impl_->workspaces.emplace(
            items.size(), std::move(workspace)).first;
    }
    return impl_->model.decode_batch(items, found->second.get(), logits, error);
}

std::size_t CudaEmbeddingBatchRunner::workspace_count() const {
    if (impl_ == nullptr) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->workspaces.size();
}

std::size_t CudaEmbeddingBatchRunner::prefill_workspace_count() const {
    if (impl_ == nullptr) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->prefill_workspaces.size();
}

std::size_t CudaEmbeddingBatchRunner::packed_prefill_count() const {
    if (impl_ == nullptr) return 0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->packed_prefill_calls;
}

}  // namespace cpp_llm_engine
