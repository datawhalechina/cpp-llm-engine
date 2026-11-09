// Engine CUDA：实现 CUDA Model/Context 到 Chapter06 RequestModelExecutor 的适配。
// 所有权：Factory 共享 Model、分页 payload 与 batch workspace；Executor 独占 Context。
// 验证：真实 Runtime CTest 检查生成、批量解码与 Radix 前缀采用。
#include "cpp_llm_engine/real_llama_cuda_executor_factory.h"

#include "cpp_llm_engine/model/real_llama_cuda_model.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "chapter06_generation/request.h"
#include "chapter07_kv_cache/block_table.h"
#include "chapter07_kv_cache/radix_prefix_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <unordered_set>

namespace cpp_llm_engine {
namespace {

class CudaPrefixBlockStore final : public chapter07_kv_cache::PrefixBlockStore {
public:
    CudaPrefixBlockStore(
        chapter07_kv_cache::KvBlockPool* pool,
        cuda::CudaPagedKvCache* payload,
        cuda::CudaStream* stream)
        : pool_(pool), payload_(payload), stream_(stream) {}

    std::size_t total_blocks() const override { return pool_->total_blocks(); }
    std::size_t free_blocks() const override { return pool_->free_blocks(); }
    std::size_t block_size() const override { return pool_->block_size(); }
    std::size_t token_count(std::size_t request_id) const override {
        return pool_->token_count(request_id);
    }
    std::vector<std::size_t> physical_blocks(
        std::size_t request_id) const override {
        return pool_->physical_blocks(request_id);
    }
    std::size_t block_ref_count(std::size_t physical_block) const override {
        return pool_->block_ref_count(physical_block);
    }
    bool attach_shared_prefix(
        std::size_t request_id,
        const std::vector<std::size_t>& physical_blocks,
        std::size_t token_count) override {
        return pool_->attach_shared_prefix(
            request_id, physical_blocks, token_count);
    }
    bool retain_prefix_blocks(
        const std::vector<std::size_t>& physical_blocks) override {
        return pool_->retain_blocks(physical_blocks);
    }
    bool release_prefix_blocks(
        const std::vector<std::size_t>& physical_blocks) override {
        if (!clear_final_owner_blocks(physical_blocks)) return false;
        return pool_->release_retained_blocks(physical_blocks);
    }

    bool release_request(std::size_t request_id) {
        if (!pool_->contains(request_id)) return true;
        const auto blocks = pool_->physical_blocks(request_id);
        if (!clear_final_owner_blocks(blocks)) return false;
        return pool_->release(request_id);
    }

private:
    bool clear_final_owner_blocks(
        const std::vector<std::size_t>& physical_blocks) {
        bool submitted = false;
        for (const std::size_t block : physical_blocks) {
            if (pool_->block_ref_count(block) != 1) continue;
            const auto status = payload_->clear_block(block, *stream_);
            if (!status.is_ok()) return false;
            submitted = true;
        }
        return !submitted || stream_->synchronize().is_ok();
    }

    chapter07_kv_cache::KvBlockPool* pool_ = nullptr;
    cuda::CudaPagedKvCache* payload_ = nullptr;
    cuda::CudaStream* stream_ = nullptr;
};

class CudaPagedKvDataPlane final : public model::RealLlamaCudaKvDataPlane {
public:
    CudaPagedKvDataPlane(
        std::size_t request_id,
        chapter07_kv_cache::KvBlockPool* pool,
        cuda::CudaPagedKvCache* payload,
        chapter07_kv_cache::RadixPrefixCache* radix_cache)
        : request_id_(request_id), pool_(pool), payload_(payload),
          radix_cache_(radix_cache) {}

    std::size_t request_id() const override { return request_id_; }
    std::size_t token_count() const override {
        return pool_->token_count(request_id_);
    }
    bool reserve(std::size_t total_tokens, std::string* error) override {
        if (pool_->reserve(request_id_, total_tokens)) {
            if (error != nullptr) error->clear();
            return true;
        }
        const std::size_t target_blocks =
            total_tokens / pool_->block_size() +
            (total_tokens % pool_->block_size() == 0 ? 0 : 1);
        const std::size_t current_blocks = pool_->block_count(request_id_);
        const std::size_t additional_blocks =
            target_blocks > current_blocks ? target_blocks - current_blocks : 0;
        if (radix_cache_ != nullptr && additional_blocks > 0 &&
            radix_cache_->evict_until_free(additional_blocks) &&
            pool_->reserve(request_id_, total_tokens)) {
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
    chapter07_kv_cache::RadixPrefixCache* radix_cache_ = nullptr;
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
                    ? "CUDA prefill requires at least one token"
                    : "CUDA prefill logits output must not be null";
            }
            return false;
        }
        if (start_new_sequence) {
            context_->reset();
        } else if (context_->position() == 0) {
            if (error != nullptr) {
                *error = "CUDA continued prefill requires an existing context";
            }
            return false;
        }
        chapter06_generation::Logits current;
        auto* output = produce_logits ? logits : &current;
        const auto status = context_->prefill_tokens(prompt, output);
        if (!status.is_ok()) {
            if (error != nullptr) *error = status.message();
            context_->reset();
            return false;
        }
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
    std::unique_ptr<CudaPrefixBlockStore> prefix_store;
    std::unique_ptr<chapter07_kv_cache::RadixPrefixCache> radix_cache;
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
    impl_->prefix_store = std::make_unique<CudaPrefixBlockStore>(
        impl_->paged_pool.get(), impl_->paged_payload.get(), &impl_->release_stream);
    impl_->radix_cache = std::make_unique<chapter07_kv_cache::RadixPrefixCache>(
        impl_->prefix_store.get());
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
            request_id, impl_->paged_pool.get(), impl_->paged_payload.get(),
            impl_->radix_cache.get()),
        error);
}

void RealLlamaCudaExecutorFactory::release_request(std::size_t request_id) const {
    if (impl_->prefix_store != nullptr) {
        (void)impl_->prefix_store->release_request(request_id);
    }
}

std::size_t RealLlamaCudaExecutorFactory::estimate_cached_prefix(
    const chapter06_generation::Request& request) const {
    if (impl_->radix_cache == nullptr || request.prompt_size() <= 1 ||
        request.prefill_offset() != 0 || !request.token_history().empty() ||
        request.terminal()) {
        return 0;
    }
    chapter06_generation::TokenSequence eligible(
        request.prompt().begin(), request.prompt().end() - 1);
    return impl_->radix_cache->estimate_prefix_tokens(eligible);
}

bool RealLlamaCudaExecutorFactory::adopt_cached_prefix(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    std::size_t request_id,
    chapter06_generation::Request& request,
    std::string* error) const {
    auto* real = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaCudaRequestExecutor*>(executor->underlying_executor());
    if (real == nullptr || impl_->paged_pool == nullptr ||
        impl_->radix_cache == nullptr) {
        if (error != nullptr) {
            *error = "CUDA Radix adoption requires a paged real Llama executor";
        }
        return false;
    }
    if (request.prompt_size() <= 1 || request.prefill_offset() != 0 ||
        !request.token_history().empty() || request.terminal()) {
        if (error != nullptr) error->clear();
        return true;
    }

    chapter06_generation::TokenSequence eligible(
        request.prompt().begin(), request.prompt().end() - 1);
    const auto match = impl_->radix_cache->longest_prefix(eligible);
    if (match.token_count == 0) {
        if (error != nullptr) error->clear();
        return true;
    }
    if (!impl_->radix_cache->attach(request_id, match)) {
        if (error != nullptr) *error = "CUDA Radix blocks cannot attach to request";
        return false;
    }
    if (!real->context()->adopt_external_kv_prefix(match.token_count, error)) {
        impl_->prefix_store->release_request(request_id);
        return false;
    }
    if (!request.adopt_prefilled_prefix(match.token_count)) {
        real->context()->reset();
        impl_->prefix_store->release_request(request_id);
        if (error != nullptr) *error = "request cannot adopt CUDA cached prefix";
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

void RealLlamaCudaExecutorFactory::publish_cached_prefix(
    std::size_t request_id,
    const chapter06_generation::Request& request) const {
    if (impl_->radix_cache == nullptr || request.prefill_offset() == 0) return;
    const std::size_t token_count = std::min(
        request.prefill_offset(), request.prompt_size());
    chapter06_generation::TokenSequence committed(
        request.prompt().begin(),
        request.prompt().begin() +
            static_cast<chapter06_generation::TokenSequence::difference_type>(token_count));
    impl_->radix_cache->publish(committed, request_id);
}

bool RealLlamaCudaExecutorFactory::prefix_cache_stats(
    RealLlamaCudaPrefixCacheStats* stats,
    std::string* error) const {
    if (stats == nullptr || impl_->radix_cache == nullptr) {
        if (error != nullptr) {
            *error = stats == nullptr
                ? "CUDA prefix cache stats output must not be null"
                : "CUDA paged prefix cache is not configured";
        }
        return false;
    }
    const auto source = impl_->radix_cache->stats();
    *stats = {
        source.queries,
        source.hits,
        source.misses,
        source.matched_tokens,
        source.evicted_blocks,
        impl_->radix_cache->cached_blocks(),
    };
    if (error != nullptr) error->clear();
    return true;
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
