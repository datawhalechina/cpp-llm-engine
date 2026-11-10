// Engine：实现 Qwen2MoE Model/Context 到 Chapter06 执行器契约的适配。
#include "cpp_llm_engine/real_qwen2_moe_executor_factory.h"

#include "cpp_llm_engine/model/real_qwen2_moe_cpu_model.h"

#include <atomic>
#include <set>
#include <utility>

namespace cpp_llm_engine {
namespace {

class RealQwen2MoeRequestExecutor final
    : public chapter06_generation::RequestModelExecutor {
public:
    explicit RealQwen2MoeRequestExecutor(
        std::unique_ptr<model::RealQwen2MoeCpuContext> context)
        : context_(std::move(context)) {}

    bool prefill(const chapter06_generation::TokenSequence& prompt,
                 chapter06_generation::Logits* logits,
                 std::string* error) override {
        context_->reset();
        return context_->prefill_tokens(prompt, logits, error);
    }

    bool supports_chunked_prefill() const override { return true; }

    bool prefill_chunk(const chapter06_generation::TokenSequence& prompt,
                       bool start_new_sequence,
                       bool produce_logits,
                       chapter06_generation::Logits* logits,
                       std::string* error) override {
        if (start_new_sequence) context_->reset();
        chapter06_generation::Logits chunk_logits;
        if (!context_->prefill_tokens(prompt, &chunk_logits, error)) return false;
        if (produce_logits) *logits = std::move(chunk_logits);
        return true;
    }

    bool decode(chapter06_generation::TokenId token,
                chapter06_generation::Logits* logits,
                std::string* error) override {
        return context_->decode_token(token, logits, error);
    }

    chapter06_generation::ModelExecutionState execution_state() const override {
        return {true, context_->position(), context_->layer_count(), true};
    }

    model::RealQwen2MoeCpuContext* context() const { return context_.get(); }

private:
    std::unique_ptr<model::RealQwen2MoeCpuContext> context_;
};

}  // namespace

struct RealQwen2MoeExecutorFactory::Impl {
    std::shared_ptr<model::RealQwen2MoeCpuModel> model;
    std::atomic<RealQwen2MoeBatchMode> batch_mode{
        RealQwen2MoeBatchMode::Grouped};
};

std::unique_ptr<RealQwen2MoeExecutorFactory>
RealQwen2MoeExecutorFactory::load(
    const std::string& model_path, std::string* error) {
    auto model = model::RealQwen2MoeCpuModel::load(model_path, error);
    if (model == nullptr) return nullptr;
    auto impl = std::make_shared<Impl>();
    impl->model = std::move(model);
    return std::unique_ptr<RealQwen2MoeExecutorFactory>(
        new RealQwen2MoeExecutorFactory(std::move(impl)));
}

RealQwen2MoeExecutorFactory::RealQwen2MoeExecutorFactory(
    std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

std::shared_ptr<chapter06_generation::RequestModelExecutor>
RealQwen2MoeExecutorFactory::create_executor() const {
    return std::make_shared<RealQwen2MoeRequestExecutor>(
        impl_->model->create_context());
}

std::size_t RealQwen2MoeExecutorFactory::layer_count() const {
    return impl_->model->metadata().layer_count;
}

std::size_t RealQwen2MoeExecutorFactory::vocab_size() const {
    return impl_->model->metadata().vocab_size;
}

chapter06_generation::TokenId
RealQwen2MoeExecutorFactory::eos_token_id() const {
    return impl_->model->metadata().eos_token_id;
}

bool RealQwen2MoeExecutorFactory::decode_batch(
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
        if (error != nullptr) *error = "Qwen2MoE batch input sizes differ";
        return false;
    }
    std::set<std::size_t> unique_ids;
    std::set<model::RealQwen2MoeCpuContext*> unique_contexts;
    std::vector<model::RealQwen2MoeCpuContext*> contexts;
    contexts.reserve(executors.size());
    for (std::size_t index = 0; index < executors.size(); ++index) {
        auto* executor = dynamic_cast<RealQwen2MoeRequestExecutor*>(
            executors[index] == nullptr
                ? nullptr
                : executors[index]->underlying_executor());
        auto* context = executor == nullptr ? nullptr : executor->context();
        if (context == nullptr || context->position() != expected_positions[index] ||
            !unique_ids.insert(request_ids[index]).second ||
            !unique_contexts.insert(context).second) {
            if (error != nullptr) {
                *error = "Qwen2MoE batch requires unique aligned requests";
            }
            return false;
        }
        contexts.push_back(context);
    }
    const auto mode = impl_->batch_mode.load(std::memory_order_relaxed) ==
            RealQwen2MoeBatchMode::Grouped
        ? model::RealMoeBatchMode::Grouped
        : model::RealMoeBatchMode::RowWise;
    return impl_->model->decode_batch(
        contexts, tokens, logits, nullptr, error, mode);
}

void RealQwen2MoeExecutorFactory::set_layer_cache_capacity(
    std::size_t layer_count) {
    impl_->model->set_layer_cache_capacity(layer_count);
}

RealQwen2MoeLayerCacheStats
RealQwen2MoeExecutorFactory::layer_cache_stats() const {
    const auto stats = impl_->model->layer_cache_stats();
    return {stats.capacity, stats.resident_layers, stats.resident_bytes,
            stats.hits, stats.misses};
}

void RealQwen2MoeExecutorFactory::set_moe_batch_mode(
    RealQwen2MoeBatchMode mode) {
    impl_->batch_mode.store(mode, std::memory_order_relaxed);
}

RealQwen2MoeBatchMode RealQwen2MoeExecutorFactory::moe_batch_mode() const {
    return impl_->batch_mode.load(std::memory_order_relaxed);
}

}  // namespace cpp_llm_engine
