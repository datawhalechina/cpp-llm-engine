// Engine：实现 DeepSeek2 Model/Context 到 Chapter06 执行器契约的适配。
#include "cpp_llm_engine/real_deepseek2_executor_factory.h"

#include "cpp_llm_engine/model/real_deepseek2_cpu_model.h"

#include <atomic>
#include <set>
#include <utility>

namespace cpp_llm_engine {
namespace {

class RealDeepSeek2RequestExecutor final
    : public chapter06_generation::RequestModelExecutor {
public:
    explicit RealDeepSeek2RequestExecutor(
        std::unique_ptr<model::RealDeepSeek2CpuContext> context)
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

    model::RealDeepSeek2CpuContext* context() const { return context_.get(); }

private:
    std::unique_ptr<model::RealDeepSeek2CpuContext> context_;
};

}  // namespace

struct RealDeepSeek2ExecutorFactory::Impl {
    std::shared_ptr<model::RealDeepSeek2CpuModel> model;
    std::atomic<RealDeepSeek2MoeBatchMode> batch_mode{
        RealDeepSeek2MoeBatchMode::Grouped};
};

std::unique_ptr<RealDeepSeek2ExecutorFactory>
RealDeepSeek2ExecutorFactory::load(
    const std::string& model_path, std::string* error) {
    auto model = model::RealDeepSeek2CpuModel::load(model_path, error);
    if (model == nullptr) return nullptr;
    auto impl = std::make_shared<Impl>();
    impl->model = std::move(model);
    return std::unique_ptr<RealDeepSeek2ExecutorFactory>(
        new RealDeepSeek2ExecutorFactory(std::move(impl)));
}

RealDeepSeek2ExecutorFactory::RealDeepSeek2ExecutorFactory(
    std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

std::shared_ptr<chapter06_generation::RequestModelExecutor>
RealDeepSeek2ExecutorFactory::create_executor() const {
    return std::make_shared<RealDeepSeek2RequestExecutor>(
        impl_->model->create_context());
}

std::size_t RealDeepSeek2ExecutorFactory::layer_count() const {
    return impl_->model->metadata().layer_count;
}

std::size_t RealDeepSeek2ExecutorFactory::vocab_size() const {
    return impl_->model->metadata().vocab_size;
}

chapter06_generation::TokenId
RealDeepSeek2ExecutorFactory::eos_token_id() const {
    return impl_->model->metadata().eos_token_id;
}

bool RealDeepSeek2ExecutorFactory::decode_batch(
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
        if (error != nullptr) *error = "DeepSeek2 batch input sizes differ";
        return false;
    }
    std::set<std::size_t> unique_ids;
    std::set<model::RealDeepSeek2CpuContext*> unique_contexts;
    std::vector<model::RealDeepSeek2CpuContext*> contexts;
    contexts.reserve(executors.size());
    for (std::size_t index = 0; index < executors.size(); ++index) {
        auto* executor = dynamic_cast<RealDeepSeek2RequestExecutor*>(
            executors[index] == nullptr
                ? nullptr
                : executors[index]->underlying_executor());
        auto* context = executor == nullptr ? nullptr : executor->context();
        if (context == nullptr || context->position() != expected_positions[index] ||
            !unique_ids.insert(request_ids[index]).second ||
            !unique_contexts.insert(context).second) {
            if (error != nullptr) {
                *error = "DeepSeek2 batch requires unique aligned requests";
            }
            return false;
        }
        contexts.push_back(context);
    }
    const auto mode = impl_->batch_mode.load(std::memory_order_relaxed) ==
            RealDeepSeek2MoeBatchMode::Grouped
        ? model::RealMoeBatchMode::Grouped
        : model::RealMoeBatchMode::RowWise;
    return impl_->model->decode_batch(contexts, tokens, logits, error, mode);
}

void RealDeepSeek2ExecutorFactory::set_layer_cache_capacity(
    std::size_t layer_count) {
    impl_->model->set_layer_cache_capacity(layer_count);
}

RealDeepSeek2LayerCacheStats
RealDeepSeek2ExecutorFactory::layer_cache_stats() const {
    const auto stats = impl_->model->layer_cache_stats();
    return {stats.capacity, stats.resident_layers, stats.resident_bytes,
            stats.hits, stats.misses};
}

void RealDeepSeek2ExecutorFactory::set_moe_batch_mode(
    RealDeepSeek2MoeBatchMode mode) {
    impl_->batch_mode.store(mode, std::memory_order_relaxed);
}

RealDeepSeek2MoeBatchMode
RealDeepSeek2ExecutorFactory::moe_batch_mode() const {
    return impl_->batch_mode.load(std::memory_order_relaxed);
}

}  // namespace cpp_llm_engine
