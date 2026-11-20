// Engine CUDA：实现请求级 DeepSeek2 CUDA Runtime 适配器。
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "cpp_llm_engine/model/real_deepseek2_cuda_model.h"
#include "chapter07_kv_cache/block_table.h"
#include "cpp_llm_engine/cuda/cuda_paged_mla_cache.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>

namespace cpp_llm_engine {
namespace {

class CudaPagedMlaDataPlane final
    : public model::RealDeepSeek2CudaMlaDataPlane {
public:
    CudaPagedMlaDataPlane(
        std::size_t request_id,
        std::shared_ptr<chapter07_kv_cache::KvBlockPool> pool,
        std::shared_ptr<cuda::CudaPagedMlaCache> payload,
        std::shared_ptr<std::mutex> mutex)
        : request_id_(request_id),
          pool_(std::move(pool)),
          payload_(std::move(payload)),
          mutex_(std::move(mutex)) {}

    std::size_t request_id() const override { return request_id_; }
    std::size_t token_count() const override {
        std::lock_guard<std::mutex> lock(*mutex_);
        return pool_->token_count(request_id_);
    }
    bool reserve(std::size_t total_tokens, std::string* error) override {
        std::lock_guard<std::mutex> lock(*mutex_);
        if (pool_->reserve(request_id_, total_tokens)) {
            if (error != nullptr) error->clear();
            return true;
        }
        if (error != nullptr) *error = "CUDA paged MLA reserve failed";
        return false;
    }
    bool commit(std::size_t total_tokens, std::string* error) override {
        std::lock_guard<std::mutex> lock(*mutex_);
        if (pool_->commit_tokens(request_id_, total_tokens)) {
            if (error != nullptr) error->clear();
            return true;
        }
        if (error != nullptr) *error = "CUDA paged MLA commit failed";
        return false;
    }
    bool trim(std::size_t total_tokens) override {
        std::lock_guard<std::mutex> lock(*mutex_);
        return pool_->trim_to_tokens(request_id_, total_tokens);
    }
    std::vector<std::size_t> physical_blocks() const override {
        std::lock_guard<std::mutex> lock(*mutex_);
        return pool_->physical_blocks(request_id_);
    }
    cuda::CudaPagedMlaCache* payload() const override { return payload_.get(); }

private:
    std::size_t request_id_ = 0;
    std::shared_ptr<chapter07_kv_cache::KvBlockPool> pool_;
    std::shared_ptr<cuda::CudaPagedMlaCache> payload_;
    std::shared_ptr<std::mutex> mutex_;
};

class RealDeepSeek2CudaRequestExecutor final
    : public chapter06_generation::RequestModelExecutor {
public:
    explicit RealDeepSeek2CudaRequestExecutor(
        std::unique_ptr<model::RealDeepSeek2CudaContext> context)
        : context_(std::move(context)) {}

    bool prefill(
        const chapter06_generation::TokenSequence& prompt,
        chapter06_generation::Logits* logits,
        std::string* error) override {
        return prefill_chunk(prompt, true, true, logits, error);
    }

    bool supports_chunked_prefill() const override { return true; }

    bool prefill_chunk(
        const chapter06_generation::TokenSequence& prompt,
        bool start_new_sequence,
        bool produce_logits,
        chapter06_generation::Logits* logits,
        std::string* error) override {
        if (prompt.empty() || (produce_logits && logits == nullptr)) {
            if (error != nullptr) {
                *error = prompt.empty()
                    ? "DeepSeek2 CUDA prefill requires at least one token"
                    : "DeepSeek2 CUDA prefill logits output must not be null";
            }
            return false;
        }
        if (start_new_sequence) {
            context_->reset();
        } else if (context_->position() == 0) {
            if (error != nullptr) {
                *error = "DeepSeek2 CUDA continued prefill requires an existing context";
            }
            return false;
        }

        chapter06_generation::Logits current;
        const auto status = context_->forward_tokens(prompt, &current);
        if (!status.is_ok()) {
            context_->reset();
            if (error != nullptr) *error = status.message();
            return false;
        }
        if (produce_logits) *logits = std::move(current);
        if (error != nullptr) error->clear();
        return true;
    }

    bool decode(
        chapter06_generation::TokenId token,
        chapter06_generation::Logits* logits,
        std::string* error) override {
        const auto status = context_->forward_token(token, logits);
        if (error != nullptr) {
            if (status.is_ok()) error->clear();
            else *error = status.message();
        }
        return status.is_ok();
    }

    chapter06_generation::ModelExecutionState execution_state() const override {
        return {true, context_->position(), context_->layer_count(), true};
    }

    model::RealDeepSeek2CudaContext* context() const { return context_.get(); }

private:
    std::unique_ptr<model::RealDeepSeek2CudaContext> context_;
};

}  // namespace

struct RealDeepSeek2CudaExecutorFactory::Impl final {
    std::shared_ptr<model::RealDeepSeek2CudaModel> model;
    std::size_t context_capacity = 0;
    cuda::CudaMlaComputePath mla_path =
        cuda::CudaMlaComputePath::LegacyThreeStage;
    mutable std::mutex batch_mutex;
    mutable std::unordered_map<
        std::size_t,
        std::unique_ptr<model::RealDeepSeek2CudaBatchWorkspace>>
        batch_workspaces;
    std::shared_ptr<std::mutex> paged_mutex = std::make_shared<std::mutex>();
    std::shared_ptr<chapter07_kv_cache::KvBlockPool> paged_pool;
    std::shared_ptr<cuda::CudaPagedMlaCache> paged_payload;
    cuda::CudaStream release_stream;
    std::set<std::size_t> bound_request_ids;
};

std::unique_ptr<RealDeepSeek2CudaExecutorFactory>
RealDeepSeek2CudaExecutorFactory::load(
    const std::string& model_path,
    std::size_t context_capacity,
    std::string* error,
    cuda::CudaMlaComputePath mla_path) {
    auto loaded = model::RealDeepSeek2CudaModel::load(model_path, 0, error);
    if (loaded == nullptr) return nullptr;
    if (context_capacity == 0 ||
        context_capacity > loaded->max_context_tokens()) {
        if (error != nullptr) {
            *error = "DeepSeek2 CUDA executor context capacity is invalid";
        }
        return nullptr;
    }
    auto impl = std::make_unique<Impl>();
    impl->model = std::shared_ptr<model::RealDeepSeek2CudaModel>(
        std::move(loaded));
    impl->context_capacity = context_capacity;
    impl->mla_path = mla_path;
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealDeepSeek2CudaExecutorFactory>(
        new RealDeepSeek2CudaExecutorFactory(std::move(impl)));
}

RealDeepSeek2CudaExecutorFactory::RealDeepSeek2CudaExecutorFactory(
    std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RealDeepSeek2CudaExecutorFactory::~RealDeepSeek2CudaExecutorFactory() = default;
RealDeepSeek2CudaExecutorFactory::RealDeepSeek2CudaExecutorFactory(
    RealDeepSeek2CudaExecutorFactory&&) noexcept = default;
RealDeepSeek2CudaExecutorFactory&
RealDeepSeek2CudaExecutorFactory::operator=(
    RealDeepSeek2CudaExecutorFactory&&) noexcept = default;

std::shared_ptr<chapter06_generation::RequestModelExecutor>
RealDeepSeek2CudaExecutorFactory::create_executor() const {
    return create_executor(impl_->mla_path);
}

std::shared_ptr<chapter06_generation::RequestModelExecutor>
RealDeepSeek2CudaExecutorFactory::create_executor(
    cuda::CudaMlaComputePath mla_path) const {
    std::string error;
    auto context = impl_->model->create_context(
        impl_->context_capacity, &error, mla_path);
    if (context == nullptr) return {};
    return std::make_shared<RealDeepSeek2CudaRequestExecutor>(
        std::move(context));
}

bool RealDeepSeek2CudaExecutorFactory::configure_paged_mla_cache(
    std::size_t total_blocks,
    std::size_t block_size,
    std::string* error) {
    if (total_blocks == 0 || block_size == 0 || impl_->paged_pool != nullptr) {
        if (error != nullptr) {
            *error = "CUDA paged MLA configuration is invalid";
        }
        return false;
    }
    auto payload = std::make_shared<cuda::CudaPagedMlaCache>();
    core::Status status = cuda::CudaPagedMlaCache::create(
        impl_->model->layer_count(),
        total_blocks,
        block_size,
        impl_->model->mla_latent_dimension(),
        impl_->model->mla_rope_dimension(),
        payload.get());
    if (status.is_ok()) {
        status = cuda::CudaStream::create(&impl_->release_stream).to_core_status();
    }
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
RealDeepSeek2CudaExecutorFactory::paged_mla_resource() const {
    return impl_->paged_pool;
}

bool RealDeepSeek2CudaExecutorFactory::bind_request(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    std::size_t request_id,
    std::string* error) const {
    auto* real = executor == nullptr ? nullptr
        : dynamic_cast<RealDeepSeek2CudaRequestExecutor*>(
            executor->underlying_executor());
    {
        std::lock_guard<std::mutex> lock(*impl_->paged_mutex);
        if (real == nullptr || impl_->paged_pool == nullptr ||
            impl_->paged_payload == nullptr ||
            !impl_->bound_request_ids.insert(request_id).second) {
            if (error != nullptr) *error = "CUDA paged MLA binding is invalid";
            return false;
        }
    }
    const bool attached = real->context()->attach_mla_data_plane(
        std::make_shared<CudaPagedMlaDataPlane>(
            request_id,
            impl_->paged_pool,
            impl_->paged_payload,
            impl_->paged_mutex),
        error);
    if (!attached) {
        std::lock_guard<std::mutex> lock(*impl_->paged_mutex);
        impl_->bound_request_ids.erase(request_id);
    }
    return attached;
}

void RealDeepSeek2CudaExecutorFactory::release_request(
    std::size_t request_id) const {
    std::lock_guard<std::mutex> lock(*impl_->paged_mutex);
    if (impl_->paged_pool == nullptr ||
        impl_->bound_request_ids.find(request_id) ==
            impl_->bound_request_ids.end()) {
        return;
    }
    if (!impl_->paged_pool->contains(request_id)) {
        impl_->bound_request_ids.erase(request_id);
        return;
    }
    const bool released = chapter07_kv_cache::release_after_payload_cleanup(
        *impl_->paged_pool,
        request_id,
        [&](std::size_t block) {
            return impl_->paged_payload
                ->clear_block(block, impl_->release_stream)
                .is_ok();
        },
        [&] { return impl_->release_stream.synchronize().is_ok(); });
    if (released) {
        impl_->bound_request_ids.erase(request_id);
    }
}

bool RealDeepSeek2CudaExecutorFactory::decode_batch(
    const std::vector<std::shared_ptr<
        chapter06_generation::RequestModelExecutor>>& executors,
    const chapter06_generation::TokenSequence& tokens,
    const std::vector<std::size_t>& expected_positions,
    const std::vector<std::size_t>& request_ids,
    std::vector<chapter06_generation::Logits>* logits,
    std::string* error) const {
    if (logits == nullptr || executors.empty() ||
        executors.size() != tokens.size() ||
        tokens.size() != expected_positions.size() ||
        tokens.size() != request_ids.size()) {
        if (error != nullptr) *error = "DeepSeek2 CUDA batch sizes differ";
        return false;
    }
    std::set<std::size_t> unique_ids;
    std::set<model::RealDeepSeek2CudaContext*> unique_contexts;
    std::vector<model::RealDeepSeek2CudaBatchItem> items;
    items.reserve(executors.size());
    for (std::size_t index = 0; index < executors.size(); ++index) {
        auto* executor = dynamic_cast<RealDeepSeek2CudaRequestExecutor*>(
            executors[index] == nullptr
                ? nullptr
                : executors[index]->underlying_executor());
        auto* context = executor == nullptr ? nullptr : executor->context();
        if (context == nullptr ||
            context->position() != expected_positions[index] ||
            !unique_ids.insert(request_ids[index]).second ||
            !unique_contexts.insert(context).second) {
            if (error != nullptr) {
                *error = "DeepSeek2 CUDA batch requires unique aligned requests";
            }
            return false;
        }
        items.push_back({context, tokens[index]});
    }

    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    auto found = impl_->batch_workspaces.find(items.size());
    if (found == impl_->batch_workspaces.end()) {
        auto workspace = impl_->model->create_batch_workspace(
            items.size(), error);
        if (workspace == nullptr) return false;
        found = impl_->batch_workspaces.emplace(
            items.size(), std::move(workspace)).first;
    }
    return impl_->model->decode_batch(
        items, found->second.get(), logits, error);
}

std::size_t RealDeepSeek2CudaExecutorFactory::layer_count() const {
    return impl_->model->layer_count();
}
std::size_t RealDeepSeek2CudaExecutorFactory::vocab_size() const {
    return impl_->model->vocab_size();
}
std::int32_t RealDeepSeek2CudaExecutorFactory::eos_token_id() const {
    return impl_->model->eos_token_id();
}
std::size_t RealDeepSeek2CudaExecutorFactory::context_capacity() const {
    return impl_->context_capacity;
}

std::size_t RealDeepSeek2CudaExecutorFactory::batch_workspace_count() const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    return impl_->batch_workspaces.size();
}
std::size_t RealDeepSeek2CudaExecutorFactory::batch_workspace_use_count(
    std::size_t batch_size) const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    const auto found = impl_->batch_workspaces.find(batch_size);
    return found == impl_->batch_workspaces.end()
        ? 0 : found->second->use_count();
}
std::size_t RealDeepSeek2CudaExecutorFactory::grouped_moe_call_count(
    std::size_t batch_size) const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    const auto found = impl_->batch_workspaces.find(batch_size);
    return found == impl_->batch_workspaces.end()
        ? 0 : found->second->grouped_moe_calls();
}
std::size_t RealDeepSeek2CudaExecutorFactory::active_expert_group_count(
    std::size_t batch_size) const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    const auto found = impl_->batch_workspaces.find(batch_size);
    return found == impl_->batch_workspaces.end()
        ? 0 : found->second->active_expert_groups();
}
std::size_t RealDeepSeek2CudaExecutorFactory::multi_row_expert_group_count(
    std::size_t batch_size) const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    const auto found = impl_->batch_workspaces.find(batch_size);
    return found == impl_->batch_workspaces.end()
        ? 0 : found->second->multi_row_expert_groups();
}
std::size_t RealDeepSeek2CudaExecutorFactory::routed_row_count(
    std::size_t batch_size) const {
    std::lock_guard<std::mutex> lock(impl_->batch_mutex);
    const auto found = impl_->batch_workspaces.find(batch_size);
    return found == impl_->batch_workspaces.end()
        ? 0 : found->second->routed_rows();
}

}  // namespace cpp_llm_engine
