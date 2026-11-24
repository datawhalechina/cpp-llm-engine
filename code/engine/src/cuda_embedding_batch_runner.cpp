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
    explicit Impl(const model::RealLlamaCudaModel& source) : model(source) {}

    const model::RealLlamaCudaModel& model;
    mutable std::mutex mutex;
    mutable std::map<std::size_t, std::unique_ptr<
        model::RealLlamaCudaBatchWorkspace>> workspaces;
};

CudaEmbeddingBatchRunner::CudaEmbeddingBatchRunner(
    const model::RealLlamaCudaModel& model)
    : impl_(std::make_unique<Impl>(model)) {}

CudaEmbeddingBatchRunner::~CudaEmbeddingBatchRunner() = default;
CudaEmbeddingBatchRunner::CudaEmbeddingBatchRunner(
    CudaEmbeddingBatchRunner&&) noexcept = default;
CudaEmbeddingBatchRunner& CudaEmbeddingBatchRunner::operator=(
    CudaEmbeddingBatchRunner&&) noexcept = default;

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

}  // namespace cpp_llm_engine
