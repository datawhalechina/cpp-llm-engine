// Engine CUDA：实现 CUDA Model/Context 到 Chapter06 RequestModelExecutor 的适配。
// 所有权：Factory 共享 Model、分页 payload 与 batch workspace；Executor 独占 Context。
// 验证：真实 Runtime CTest 检查多 token 生成、重复请求和状态推进。
#include "cpp_llm_engine/real_llama_cuda_executor_factory.h"

#include "cpp_llm_engine/model/real_llama_cuda_model.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "chapter07_kv_cache/block_table.h"
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <mutex>
#include <unordered_map>
#include <utility>
#include <unordered_set>

namespace cpp_llm_engine {
namespace {

class CudaPagedKvDataPlane final : public model::RealLlamaCudaKvDataPlane {
public:
    CudaPagedKvDataPlane(
        std::size_t request_id,
        chapter07_kv_cache::KvBlockPool* pool,
        cuda::CudaPagedKvCache* payload)
        : request_id_(request_id), pool_(pool), payload_(payload) {}

    std::size_t request_id() const override { return request_id_; }
    std::size_t token_count() const override {
        return pool_->token_count(request_id_);
    }
    bool reserve(std::size_t total_tokens, std::string* error) override {
        if (pool_->reserve(request_id_, total_tokens)) {
            if (error != nullptr) error->clear();
            return true;
        }
        if (error != nullptr) *error = "CUDA paged KV reserve failed";
        return false;
    }
    bool commit(std::size_t total_tokens, std::string* error) override {
        if (pool_->commit_tokens(request_id_, total_tokens)) {
            if (error != nullptr) error->clear();
            return true;
        }
        if (error != nullptr) *error = "CUDA paged KV commit failed";
        return false;
    }
    bool trim(std::size_t total_tokens) override {
        return pool_->trim_to_tokens(request_id_, total_tokens);
    }
    std::vector<std::size_t> physical_blocks() const override {
        return pool_->physical_blocks(request_id_);
    }
    cuda::CudaPagedKvCache* payload() const override { return payload_; }

private:
    std::size_t request_id_ = 0;
    chapter07_kv_cache::KvBlockPool* pool_ = nullptr;
    cuda::CudaPagedKvCache* payload_ = nullptr;
};

class RealLlamaCudaRequestExecutor final
    : public chapter06_generation::RequestModelExecutor {
public:
    explicit RealLlamaCudaRequestExecutor(
        std::unique_ptr<model::RealLlamaCudaContext> context,
        std::size_t layer_count)
        : context_(std::move(context)), layer_count_(layer_count) {}

    bool prefill(
        const chapter06_generation::TokenSequence& prompt,
        chapter06_generation::Logits* logits,
        std::string* error) override {
        if (prompt.empty() || logits == nullptr) {
            if (error != nullptr) *error = "CUDA prefill requires prompt and logits";
            return false;
        }
        context_->reset();
        if (!context_->uses_external_kv()) {
            const auto status = context_->prefill_tokens(prompt, logits);
            if (!status.is_ok()) {
                if (error != nullptr) *error = status.message();
                context_->reset();
                return false;
            }
            if (error != nullptr) error->clear();
            return true;
        }
        chapter06_generation::Logits current;
        for (const auto token : prompt) {
            const auto status = context_->forward_token(token, &current);
            if (!status.is_ok()) {
                if (error != nullptr) *error = status.message();
                context_->reset();
                return false;
            }
        }
        *logits = std::move(current);
        if (error != nullptr) error->clear();
        return true;
    }

    bool decode(
        chapter06_generation::TokenId token,
        chapter06_generation::Logits* logits,
        std::string* error) override {
        const auto status = context_->forward_token(token, logits);
        if (!status.is_ok() && error != nullptr) *error = status.message();
        if (status.is_ok() && error != nullptr) error->clear();
        return status.is_ok();
    }

    chapter06_generation::ModelExecutionState execution_state() const override {
        return {true, context_->position(), layer_count_, true};
    }

    model::RealLlamaCudaContext* context() const { return context_.get(); }

private:
    std::unique_ptr<model::RealLlamaCudaContext> context_;
    std::size_t layer_count_ = 0;
};

}  // namespace

struct RealLlamaCudaExecutorFactory::Impl final {
    std::shared_ptr<model::RealLlamaCudaModel> model;
    std::size_t context_capacity = 0;
    mutable std::mutex batch_mutex;
    mutable std::unordered_map<
        std::size_t,
        std::unique_ptr<model::RealLlamaCudaBatchWorkspace>> batch_workspaces;
    mutable std::size_t batch_workspace_creation_count = 0;
    std::shared_ptr<chapter07_kv_cache::KvBlockPool> paged_pool;
    std::unique_ptr<cuda::CudaPagedKvCache> paged_payload;
    cuda::CudaStream release_stream;
    CudaAttentionMode attention_mode = CudaAttentionMode::ThreeStage;
};

std::unique_ptr<RealLlamaCudaExecutorFactory>
RealLlamaCudaExecutorFactory::load(
    const std::string& model_path,
    std::size_t context_capacity,
    std::string* error) {
    auto loaded = model::RealLlamaCudaModel::load(model_path, 0, error);
    if (loaded == nullptr) return nullptr;
    if (context_capacity == 0 ||
        context_capacity > loaded->max_context_tokens()) {
        if (error != nullptr) *error = "CUDA executor context capacity is invalid";
        return nullptr;
    }
    auto impl = std::make_unique<Impl>();
    impl->model = std::shared_ptr<model::RealLlamaCudaModel>(std::move(loaded));
    impl->context_capacity = context_capacity;
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaCudaExecutorFactory>(
        new RealLlamaCudaExecutorFactory(std::move(impl)));
}

RealLlamaCudaExecutorFactory::RealLlamaCudaExecutorFactory(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCudaExecutorFactory::~RealLlamaCudaExecutorFactory() = default;
RealLlamaCudaExecutorFactory::RealLlamaCudaExecutorFactory(
    RealLlamaCudaExecutorFactory&&) noexcept = default;
RealLlamaCudaExecutorFactory& RealLlamaCudaExecutorFactory::operator=(
    RealLlamaCudaExecutorFactory&&) noexcept = default;

std::shared_ptr<chapter06_generation::RequestModelExecutor>
RealLlamaCudaExecutorFactory::create_executor() const {
    std::string error;
    auto context = impl_->model->create_context(impl_->context_capacity, &error);
    if (context == nullptr) return {};
    context->set_attention_path(
        impl_->attention_mode == CudaAttentionMode::FusedOnline
            ? cuda::CudaAttentionPath::FusedOnline
            : cuda::CudaAttentionPath::ThreeStage);
    return std::make_shared<RealLlamaCudaRequestExecutor>(
        std::move(context), impl_->model->layer_count());
}

void RealLlamaCudaExecutorFactory::set_attention_mode(CudaAttentionMode mode) {
    impl_->attention_mode = mode;
}

CudaAttentionMode RealLlamaCudaExecutorFactory::attention_mode() const {
    return impl_->attention_mode;
}

bool RealLlamaCudaExecutorFactory::configure_paged_kv_cache(
    std::size_t total_blocks,
    std::size_t block_size,
    std::string* error) {
    if (total_blocks == 0 || block_size == 0 || impl_->paged_pool != nullptr) {
        if (error != nullptr) *error = "CUDA paged KV configuration is invalid";
        return false;
    }
    auto payload = std::make_unique<cuda::CudaPagedKvCache>();
    core::Status status = cuda::CudaPagedKvCache::create(
        impl_->model->layer_count(), total_blocks, block_size,
        impl_->model->kv_head_count(), impl_->model->head_dimension(), payload.get());
    if (status.is_ok()) status = cuda::CudaStream::create(&impl_->release_stream).to_core_status();
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return false;
    }
    impl_->paged_pool = std::make_shared<chapter07_kv_cache::KvBlockPool>(
        total_blocks, block_size);
    impl_->paged_payload = std::move(payload);
    if (error != nullptr) error->clear();
    return true;
}

std::shared_ptr<chapter07_kv_cache::KvCacheTransaction>
RealLlamaCudaExecutorFactory::paged_kv_resource() const {
    return impl_->paged_pool;
}

bool RealLlamaCudaExecutorFactory::bind_request(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    std::size_t request_id,
    std::string* error) const {
    auto* real = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaCudaRequestExecutor*>(executor->underlying_executor());
    if (real == nullptr || impl_->paged_pool == nullptr || impl_->paged_payload == nullptr ||
        impl_->paged_pool->contains(request_id)) {
        if (error != nullptr) *error = "CUDA paged request binding is invalid";
        return false;
    }
    return real->context()->attach_kv_data_plane(
        std::make_shared<CudaPagedKvDataPlane>(
            request_id, impl_->paged_pool.get(), impl_->paged_payload.get()),
        error);
}

void RealLlamaCudaExecutorFactory::release_request(std::size_t request_id) const {
    if (impl_->paged_pool == nullptr || impl_->paged_payload == nullptr ||
        !impl_->paged_pool->contains(request_id)) return;
    const auto blocks = impl_->paged_pool->physical_blocks(request_id);
    for (const std::size_t block : blocks) {
        (void)impl_->paged_payload->clear_block(block, impl_->release_stream);
    }
    (void)impl_->release_stream.synchronize();
    (void)impl_->paged_pool->release(request_id);
}

bool RealLlamaCudaExecutorFactory::decode_batch(
    const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>& executors,
    const chapter06_generation::TokenSequence& tokens,
    const std::vector<std::size_t>& expected_positions,
    const std::vector<std::size_t>& request_ids,
    std::vector<chapter06_generation::Logits>* logits,
    std::string* error) const {
    if (executors.empty() || executors.size() != tokens.size() ||
        executors.size() != expected_positions.size() ||
        executors.size() != request_ids.size() || logits == nullptr) {
        if (error != nullptr) *error = "CUDA decode batch input sizes do not match";
        return false;
    }
    std::unordered_set<const RealLlamaCudaRequestExecutor*> unique_executors;
    std::unordered_set<std::size_t> unique_request_ids;
    std::vector<model::RealLlamaCudaBatchItem> items;
    items.reserve(executors.size());
    for (std::size_t index = 0; index < executors.size(); ++index) {
        auto* executor = executors[index] == nullptr ? nullptr
            : dynamic_cast<RealLlamaCudaRequestExecutor*>(
                executors[index]->underlying_executor());
        if (executor == nullptr || !unique_executors.insert(executor).second ||
            !unique_request_ids.insert(request_ids[index]).second) {
            if (error != nullptr) {
                *error = "CUDA decode batch requires unique executors and request IDs";
            }
            return false;
        }
        const auto state = executor->execution_state();
        if (!state.available || state.committed_tokens == 0 ||
            state.committed_tokens != expected_positions[index]) {
            if (error != nullptr) *error = "CUDA decode batch position mismatch";
            return false;
        }
        if (executor->context()->uses_external_kv()) {
            if (!executor->context()->uses_external_kv_for(request_ids[index]) ||
                impl_->paged_pool == nullptr ||
                !impl_->paged_pool->contains(request_ids[index]) ||
                impl_->paged_pool->token_count(request_ids[index]) !=
                    expected_positions[index] ||
                impl_->paged_pool->capacity_tokens(request_ids[index]) <=
                    expected_positions[index]) {
                if (error != nullptr) {
                    *error = "CUDA paged batch request is not bound, aligned, or reserved";
                }
                return false;
            }
        }
        items.push_back({executor->context(), tokens[index]});
    }
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    auto found = impl_->batch_workspaces.find(items.size());
    if (found == impl_->batch_workspaces.end()) {
        auto workspace = impl_->model->create_batch_workspace(items.size(), error);
        if (workspace == nullptr) return false;
        found = impl_->batch_workspaces.emplace(
            items.size(), std::move(workspace)).first;
        ++impl_->batch_workspace_creation_count;
    }
    return impl_->model->decode_batch(items, found->second.get(), logits, error);
}

std::size_t RealLlamaCudaExecutorFactory::vocab_size() const {
    return impl_->model->vocab_size();
}
std::size_t RealLlamaCudaExecutorFactory::layer_count() const {
    return impl_->model->layer_count();
}
std::int32_t RealLlamaCudaExecutorFactory::eos_token_id() const {
    return impl_->model->eos_token_id();
}
std::size_t RealLlamaCudaExecutorFactory::context_capacity() const {
    return impl_->context_capacity;
}

std::size_t RealLlamaCudaExecutorFactory::batch_workspace_count() const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    return impl_->batch_workspaces.size();
}

std::size_t RealLlamaCudaExecutorFactory::batch_workspace_creation_count() const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    return impl_->batch_workspace_creation_count;
}

std::size_t RealLlamaCudaExecutorFactory::batch_workspace_use_count(
    std::size_t batch_size) const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    const auto found = impl_->batch_workspaces.find(batch_size);
    return found == impl_->batch_workspaces.end() || found->second == nullptr
        ? 0 : found->second->use_count();
}

std::size_t RealLlamaCudaExecutorFactory::batch_workspace_device_bytes(
    std::size_t batch_size) const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    const auto found = impl_->batch_workspaces.find(batch_size);
    return found == impl_->batch_workspaces.end() || found->second == nullptr
        ? 0 : found->second->device_bytes();
}

std::size_t RealLlamaCudaExecutorFactory::batch_workspace_dynamic_allocation_count(
    std::size_t batch_size) const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    const auto found = impl_->batch_workspaces.find(batch_size);
    return found == impl_->batch_workspaces.end() || found->second == nullptr
        ? 0 : found->second->dynamic_allocation_count();
}

}  // namespace cpp_llm_engine
