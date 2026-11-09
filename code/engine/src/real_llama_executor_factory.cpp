// Engine：实现真实 Llama/Qwen3 CPU Model 到 RequestModelExecutor 的适配。
// 所有权：Factory 持有 Model；每个 Executor 持有独立 Context 及其分层 KV。
// 验证：真实生成和真实服务 CTest 共同覆盖 prefill/decode 与请求隔离。
#include "cpp_llm_engine/real_llama_executor_factory.h"

#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"
#include "chapter06_generation/request.h"
#include "chapter07_kv_cache/paged_kv_cache.h"
#include "chapter07_kv_cache/radix_prefix_cache.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace cpp_llm_engine {
namespace {

RealLlamaExecutorProfile copy_profile(const chapter01::RealLlamaDecoderProfile& source) {
    return {
        source.forward_tokens,
        source.total_ms,
        source.embedding_ms,
        source.normalization_ms,
        source.qkv_projection_ms,
        source.rope_ms,
        source.attention_ms,
        source.attention_output_ms,
        source.ffn_ms,
        source.ffn_gate_ms,
        source.ffn_up_ms,
        source.ffn_activation_ms,
        source.ffn_down_ms,
        source.ffn_residual_ms,
        source.lm_head_ms,
        source.batch_linear_calls,
        source.batch_input_pack_ms,
        source.batch_kernel_ms,
        source.batch_output_unpack_ms,
    };
}

bool configure_context(
    chapter01::RealLlamaCpuContext* context,
    RealLlamaExecutionMode mode,
    std::string* error) {
    const auto scalar = chapter05_quantization::QuantizedLinearBackendKind::ScalarFused;
    const auto q4_backend = mode == RealLlamaExecutionMode::OptimizedAvx2
        ? chapter05_quantization::QuantizedLinearBackendKind::Avx2Fused : scalar;
    const auto q6_backend = mode == RealLlamaExecutionMode::OptimizedAvx2
        ? chapter05_quantization::QuantizedLinearBackendKind::Avx2Fused : scalar;
    const chapter05_quantization::QuantizedLinearPolicy attention_policy{
        q4_backend, scalar};
    const chapter05_quantization::QuantizedLinearPolicy ffn_policy{
        q4_backend, q6_backend};
    return context->set_attention_linear_policy(attention_policy, error) &&
        context->set_ffn_linear_policy(ffn_policy, error);
}

bool paged_shape_matches(
    const cpp_llm_engine::core::DecoderModelContract& contract,
    const chapter07_kv_cache::CpuPagedKvCache& cache,
    std::string* error) {
    if (cache.layer_count() != contract.layer_count ||
        cache.kv_heads() != contract.layer.kv_head_count ||
        cache.head_dim() != contract.layer.head_dimension) {
        if (error != nullptr) *error = "paged KV shape does not match real model";
        return false;
    }
    return true;
}

class PagedKvDataPlane final : public chapter01::RealLlamaKvDataPlane {
public:
    PagedKvDataPlane(
        std::size_t request_id,
        chapter07_kv_cache::CpuPagedKvCache* cache,
        chapter07_kv_cache::RadixPrefixCache* radix_cache = nullptr)
        : request_id_(request_id), cache_(cache), radix_cache_(radix_cache) {}

    std::size_t token_count() const override {
        return cache_->token_count(request_id_);
    }

    bool reserve(std::size_t total_tokens, std::string* error) override {
        if (cache_->reserve(request_id_, total_tokens)) return true;
        const std::size_t target_blocks =
            total_tokens / cache_->block_size() +
            (total_tokens % cache_->block_size() == 0 ? 0 : 1);
        const std::size_t current_blocks = cache_->block_count(request_id_);
        const std::size_t additional_blocks =
            target_blocks > current_blocks ? target_blocks - current_blocks : 0;
        if (radix_cache_ != nullptr && additional_blocks > 0 &&
            radix_cache_->evict_until_free(additional_blocks) &&
            cache_->reserve(request_id_, total_tokens)) {
            return true;
        }
        if (error != nullptr) *error = "paged KV data plane cannot reserve token";
        return false;
    }

    bool write_token(
        std::size_t layer,
        std::size_t token_index,
        const float* key,
        const float* value,
        std::size_t element_count,
        std::string* error) override {
        return cache_->write_token(
            request_id_, layer, token_index, key, value, element_count, error);
    }

    const float* key_data(
        std::size_t layer,
        std::size_t token_index,
        std::size_t kv_head) const override {
        return cache_->key_data_reserved(request_id_, layer, token_index, kv_head);
    }

    const float* value_data(
        std::size_t layer,
        std::size_t token_index,
        std::size_t kv_head) const override {
        return cache_->value_data_reserved(request_id_, layer, token_index, kv_head);
    }

    bool commit(std::size_t total_tokens, std::string* error) override {
        return cache_->commit_tokens(request_id_, total_tokens, error);
    }

    bool trim(std::size_t total_tokens) override {
        return cache_->trim_to_tokens(request_id_, total_tokens);
    }

private:
    std::size_t request_id_;
    chapter07_kv_cache::CpuPagedKvCache* cache_;
    chapter07_kv_cache::RadixPrefixCache* radix_cache_;
};

class RealLlamaRequestExecutor final
    : public chapter06_generation::RequestModelExecutor {
public:
    explicit RealLlamaRequestExecutor(
        std::unique_ptr<chapter01::RealLlamaCpuContext> context)
        : context_(std::move(context)) {}

    chapter01::RealLlamaCpuContext* context() const { return context_.get(); }

    bool prefill(const chapter06_generation::TokenSequence& prompt,
                 chapter06_generation::Logits* logits,
                 std::string* error) override {
        const auto status = context_->prefill_tokens(prompt, logits);
        if (!status.is_ok() && error != nullptr) *error = status.message();
        return status.is_ok();
    }

    bool supports_chunked_prefill() const override { return true; }

    bool prefill_chunk(const chapter06_generation::TokenSequence& prompt,
                       bool start_new_sequence,
                       bool produce_logits,
                       chapter06_generation::Logits* logits,
                       std::string* error) override {
        const auto status = context_->prefill_chunk(
            prompt, start_new_sequence, produce_logits, logits);
        if (!status.is_ok() && error != nullptr) *error = status.message();
        return status.is_ok();
    }

    bool decode(chapter06_generation::TokenId token,
                chapter06_generation::Logits* logits,
                std::string* error) override {
        const auto status = context_->decode_token(token, logits);
        if (!status.is_ok() && error != nullptr) *error = status.message();
        return status.is_ok();
    }

    chapter06_generation::ModelExecutionState execution_state() const override {
        const auto contract = context_->model_contract();
        return {
            true,
            context_->token_count(),
            contract.layer_count,
            contract.layer.uses_kv_cache,
        };
    }

private:
    std::unique_ptr<chapter01::RealLlamaCpuContext> context_;
};

}  // namespace

struct RealLlamaExecutorFactory::Impl {
    std::unique_ptr<chapter01::RealLlamaCpuModel> model;
    std::shared_ptr<chapter07_kv_cache::CpuPagedKvCache> paged_cache;
    std::unique_ptr<chapter07_kv_cache::RadixPrefixCache> radix_cache;
    RealLlamaExecutionMode mode = RealLlamaExecutionMode::Scalar;
};

std::unique_ptr<RealLlamaExecutorFactory> RealLlamaExecutorFactory::load(
    const std::string& model_path,
    std::string* error) {
    return load(model_path, RealLlamaExecutionMode::Scalar, error);
}

std::unique_ptr<RealLlamaExecutorFactory> RealLlamaExecutorFactory::load(
    const std::string& model_path,
    RealLlamaExecutionMode mode,
    std::string* error) {
    auto model = chapter01::RealLlamaCpuModel::load(model_path, 0, error);
    if (model == nullptr) return nullptr;
    auto validation_context = model->create_context();
    if (!configure_context(validation_context.get(), mode, error)) return nullptr;
    auto impl = std::make_unique<Impl>();
    impl->model = std::move(model);
    impl->mode = mode;
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaExecutorFactory>(
        new RealLlamaExecutorFactory(std::move(impl)));
}

RealLlamaExecutorFactory::RealLlamaExecutorFactory(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaExecutorFactory::~RealLlamaExecutorFactory() = default;
RealLlamaExecutorFactory::RealLlamaExecutorFactory(
    RealLlamaExecutorFactory&&) noexcept = default;
RealLlamaExecutorFactory& RealLlamaExecutorFactory::operator=(
    RealLlamaExecutorFactory&&) noexcept = default;

std::shared_ptr<chapter06_generation::RequestModelExecutor>
RealLlamaExecutorFactory::create_executor() const {
    auto context = impl_->model->create_context();
    std::string error;
    if (!configure_context(context.get(), impl_->mode, &error)) return nullptr;
    return std::make_shared<RealLlamaRequestExecutor>(std::move(context));
}

bool RealLlamaExecutorFactory::configure_paged_kv_cache(
    std::size_t total_blocks,
    std::size_t block_size,
    std::string* error) {
    if (total_blocks == 0 || block_size == 0) {
        if (error != nullptr) *error = "paged KV capacity must be positive";
        return false;
    }
    const auto contract = decoder_contract();
    impl_->radix_cache.reset();
    impl_->paged_cache = std::make_shared<chapter07_kv_cache::CpuPagedKvCache>(
        contract.layer_count,
        total_blocks,
        block_size,
        contract.layer.kv_head_count,
        contract.layer.head_dimension);
    impl_->radix_cache = std::make_unique<chapter07_kv_cache::RadixPrefixCache>(
        impl_->paged_cache.get());
    if (error != nullptr) error->clear();
    return true;
}

std::shared_ptr<chapter07_kv_cache::KvCacheTransaction>
RealLlamaExecutorFactory::paged_kv_resource() const {
    if (impl_->paged_cache == nullptr) return {};
    return impl_->paged_cache;
}

bool RealLlamaExecutorFactory::bind_request(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    std::size_t request_id,
    std::string* error) const {
    auto* real_executor = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaRequestExecutor*>(executor->underlying_executor());
    if (real_executor == nullptr || impl_->paged_cache == nullptr) {
        if (error != nullptr) {
            *error = real_executor == nullptr
                ? "paged request binding requires a real Llama executor"
                : "paged KV cache is not configured";
        }
        return false;
    }
    return real_executor->context()->attach_kv_data_plane(
        std::make_shared<PagedKvDataPlane>(
            request_id, impl_->paged_cache.get(), impl_->radix_cache.get()), error);
}

void RealLlamaExecutorFactory::release_request(std::size_t request_id) const {
    if (impl_->paged_cache != nullptr) impl_->paged_cache->release(request_id);
}

std::size_t RealLlamaExecutorFactory::estimate_cached_prefix(
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

bool RealLlamaExecutorFactory::adopt_cached_prefix(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    std::size_t request_id,
    chapter06_generation::Request& request,
    std::string* error) const {
    auto* real_executor = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaRequestExecutor*>(executor->underlying_executor());
    if (real_executor == nullptr || impl_->paged_cache == nullptr ||
        impl_->radix_cache == nullptr) {
        if (error != nullptr) *error = "Radix adoption requires a paged real Llama executor";
        return false;
    }
    if (request.prompt_size() <= 1 || request.prefill_offset() != 0 ||
        !request.token_history().empty() || request.terminal()) {
        if (error != nullptr) error->clear();
        return true;
    }

    // K/V payload 不保存末次前向的 logits，因此至少保留一个提示词词元重新计算。
    chapter06_generation::TokenSequence eligible(
        request.prompt().begin(), request.prompt().end() - 1);
    const auto match = impl_->radix_cache->longest_prefix(eligible);
    if (match.token_count == 0) {
        if (error != nullptr) error->clear();
        return true;
    }
    if (!impl_->radix_cache->attach(request_id, match)) {
        if (error != nullptr) *error = "Radix prefix blocks cannot attach to request";
        return false;
    }
    if (!real_executor->context()->adopt_external_kv_prefix(match.token_count, error)) {
        impl_->paged_cache->release(request_id);
        return false;
    }
    if (!request.adopt_prefilled_prefix(match.token_count)) {
        real_executor->context()->reset();
        impl_->paged_cache->release(request_id);
        if (error != nullptr) *error = "request state cannot adopt cached prompt prefix";
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

void RealLlamaExecutorFactory::publish_cached_prefix(
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

bool RealLlamaExecutorFactory::prefix_cache_stats(
    RealLlamaPrefixCacheStats* stats,
    std::string* error) const {
    if (stats == nullptr || impl_->radix_cache == nullptr) {
        if (error != nullptr) {
            *error = stats == nullptr
                ? "prefix cache stats output must not be null"
                : "paged prefix cache is not configured";
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

bool RealLlamaExecutorFactory::decode_batch(
    const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>& executors,
    const chapter06_generation::TokenSequence& tokens,
    const std::vector<std::size_t>& expected_positions,
    const std::vector<std::size_t>& request_ids,
    std::vector<chapter06_generation::Logits>* logits,
    std::string* error) const {
    if (request_ids.size() != executors.size()) {
        if (error != nullptr) *error = "decode batch request IDs do not match executors";
        return false;
    }
    if (impl_->paged_cache != nullptr) {
        for (std::size_t index = 0; index < executors.size(); ++index) {
            auto* real_executor = executors[index] == nullptr ? nullptr
                : dynamic_cast<RealLlamaRequestExecutor*>(
                    executors[index]->underlying_executor());
            if (real_executor == nullptr || !real_executor->context()->uses_external_kv() ||
                impl_->paged_cache->token_count(request_ids[index]) !=
                    expected_positions[index]) {
                if (error != nullptr) *error = "paged batch request is not bound or aligned";
                return false;
            }
        }
    }
    return decode_batch(executors, tokens, expected_positions, logits, error);
}

bool RealLlamaExecutorFactory::decode_batch(
    const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>& executors,
    const chapter06_generation::TokenSequence& tokens,
    const std::vector<std::size_t>& expected_positions,
    std::vector<chapter06_generation::Logits>* logits,
    std::string* error) const {
    if (executors.empty() || executors.size() != tokens.size() ||
        executors.size() != expected_positions.size() || logits == nullptr) {
        if (error != nullptr) *error = "real decode batch input sizes do not match";
        return false;
    }

    std::vector<chapter01::RealLlamaCpuBatchItem> items;
    items.reserve(executors.size());
    std::unordered_set<const RealLlamaRequestExecutor*> unique_executors;
    for (std::size_t index = 0; index < executors.size(); ++index) {
        auto* executor = executors[index] == nullptr ? nullptr
            : dynamic_cast<RealLlamaRequestExecutor*>(
                executors[index]->underlying_executor());
        if (executor == nullptr || !unique_executors.insert(executor).second) {
            if (error != nullptr) *error = "decode batch requires unique real Llama executors";
            return false;
        }
        const auto state = executor->execution_state();
        if (!state.available || state.committed_tokens == 0 ||
            state.committed_tokens != expected_positions[index]) {
            if (error != nullptr) *error = "decode batch position does not match executor state";
            return false;
        }
        items.push_back({executor->context(), tokens[index]});
    }
    return impl_->model->decode_batch(items, logits, error);
}

bool RealLlamaExecutorFactory::sync_paged_kv(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    std::size_t request_id,
    chapter07_kv_cache::CpuPagedKvCache* cache,
    std::string* error) const {
    auto* real_executor = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaRequestExecutor*>(executor->underlying_executor());
    if (real_executor == nullptr || cache == nullptr ||
        !paged_shape_matches(decoder_contract(), *cache, error)) {
        if (real_executor == nullptr && error != nullptr) {
            *error = "paged KV sync requires a real Llama executor";
        }
        return false;
    }
    const std::size_t target_tokens = real_executor->context()->token_count();
    const std::size_t committed_tokens = cache->token_count(request_id);
    if (committed_tokens > target_tokens) {
        if (error != nullptr) *error = "paged KV is ahead of real model context";
        return false;
    }
    if (committed_tokens == target_tokens) {
        if (!real_executor->context()->uses_external_kv() &&
            !real_executor->context()->attach_kv_data_plane(
                std::make_shared<PagedKvDataPlane>(request_id, cache), error)) {
            return false;
        }
        if (error != nullptr) error->clear();
        return true;
    }
    if (!cache->reserve(request_id, target_tokens)) {
        if (error != nullptr) *error = "paged KV capacity cannot mirror real context";
        return false;
    }

    std::vector<float> key;
    std::vector<float> value;
    for (std::size_t token = committed_tokens; token < target_tokens; ++token) {
        for (std::size_t layer = 0; layer < cache->layer_count(); ++layer) {
            if (!real_executor->context()->copy_kv_token(
                    layer, token, &key, &value, error) ||
                !cache->write_token(
                    request_id, layer, token, key.data(), value.data(), key.size(), error)) {
                cache->trim_to_tokens(request_id, committed_tokens);
                return false;
            }
        }
    }
    if (!cache->commit_tokens(request_id, target_tokens, error)) {
        cache->trim_to_tokens(request_id, committed_tokens);
        return false;
    }
    if (!real_executor->context()->attach_kv_data_plane(
            std::make_shared<PagedKvDataPlane>(request_id, cache), error)) {
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaExecutorFactory::decode_batch_paged(
    const std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>>& executors,
    const chapter06_generation::TokenSequence& tokens,
    const std::vector<std::size_t>& expected_positions,
    const std::vector<std::size_t>& request_ids,
    chapter07_kv_cache::CpuPagedKvCache* cache,
    std::vector<chapter06_generation::Logits>* logits,
    std::string* error) const {
    if (cache == nullptr || executors.empty() || executors.size() != request_ids.size() ||
        executors.size() != expected_positions.size() || executors.size() != tokens.size() ||
        !paged_shape_matches(decoder_contract(), *cache, error)) {
        if (cache == nullptr && error != nullptr) *error = "paged KV output must not be null";
        return false;
    }
    std::unordered_set<std::size_t> unique_request_ids;
    for (std::size_t index = 0; index < executors.size(); ++index) {
        if (!unique_request_ids.insert(request_ids[index]).second ||
            cache->token_count(request_ids[index]) != expected_positions[index]) {
            if (error != nullptr) *error = "paged KV position does not match batch input";
            return false;
        }
    }
    if (!decode_batch(executors, tokens, expected_positions, logits, error)) {
        return false;
    }
    for (std::size_t index = 0; index < executors.size(); ++index) {
        auto* real_executor = executors[index] == nullptr ? nullptr
            : dynamic_cast<RealLlamaRequestExecutor*>(
                executors[index]->underlying_executor());
        if (real_executor == nullptr || !real_executor->context()->uses_external_kv() ||
            cache->token_count(request_ids[index]) !=
                expected_positions[index] + 1) {
            if (error != nullptr) *error = "paged decode did not commit external KV";
            return false;
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaExecutorFactory::set_executor_profiling(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    bool enabled,
    std::string* error) const {
    auto* real_executor = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaRequestExecutor*>(executor->underlying_executor());
    if (real_executor == nullptr) {
        if (error != nullptr) *error = "profiling requires a real Llama executor";
        return false;
    }
    real_executor->context()->set_profiling_enabled(enabled);
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaExecutorFactory::executor_profile(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    RealLlamaExecutorProfile* profile,
    std::string* error) const {
    auto* real_executor = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaRequestExecutor*>(executor->underlying_executor());
    if (real_executor == nullptr || profile == nullptr) {
        if (error != nullptr) *error = "profile output requires a real Llama executor";
        return false;
    }
    *profile = copy_profile(real_executor->context()->profile());
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaExecutorFactory::executor_memory(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    RealLlamaExecutorMemory* memory,
    std::string* error) const {
    auto* real_executor = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaRequestExecutor*>(executor->underlying_executor());
    if (real_executor == nullptr || memory == nullptr) {
        if (error != nullptr) *error = "memory output requires a real Llama executor";
        return false;
    }
    memory->committed_tokens = real_executor->context()->token_count();
    memory->kv_cache_bytes = real_executor->context()->kv_cache_bytes();
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaExecutorFactory::executor_uses_paged_kv(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor) const {
    auto* real_executor = executor == nullptr ? nullptr
        : dynamic_cast<RealLlamaRequestExecutor*>(executor->underlying_executor());
    return real_executor != nullptr && real_executor->context()->uses_external_kv();
}

std::size_t RealLlamaExecutorFactory::model_weight_payload_bytes() const {
    return impl_->model->weight_payload_bytes();
}

std::string RealLlamaExecutorFactory::architecture() const {
    return impl_->model->architecture();
}

std::size_t RealLlamaExecutorFactory::vocab_size() const {
    return impl_->model->vocab_size();
}

chapter06_generation::TokenId RealLlamaExecutorFactory::eos_token_id() const {
    return impl_->model->eos_token_id();
}

std::size_t RealLlamaExecutorFactory::layer_count() const {
    return impl_->model->layer_count();
}

cpp_llm_engine::core::DecoderModelContract
RealLlamaExecutorFactory::decoder_contract() const {
    return impl_->model->decoder_contract();
}

RealLlamaExecutionMode RealLlamaExecutorFactory::execution_mode() const {
    return impl_->mode;
}

}  // namespace cpp_llm_engine
