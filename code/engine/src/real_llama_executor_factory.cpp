// Engine：实现真实 TinyLlama Model 到 RequestModelExecutor 的适配。
// 所有权：Factory 持有 Model；每个 Executor 持有独立 Context 及其分层 KV。
// 验证：真实生成和真实服务 CTest 共同覆盖 prefill/decode 与请求隔离。
#include "cpp_llm_engine/real_llama_executor_factory.h"

#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"

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
        auto* executor = dynamic_cast<RealLlamaRequestExecutor*>(executors[index].get());
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

bool RealLlamaExecutorFactory::set_executor_profiling(
    const std::shared_ptr<chapter06_generation::RequestModelExecutor>& executor,
    bool enabled,
    std::string* error) const {
    auto* real_executor = dynamic_cast<RealLlamaRequestExecutor*>(executor.get());
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
    auto* real_executor = dynamic_cast<RealLlamaRequestExecutor*>(executor.get());
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
    auto* real_executor = dynamic_cast<RealLlamaRequestExecutor*>(executor.get());
    if (real_executor == nullptr || memory == nullptr) {
        if (error != nullptr) *error = "memory output requires a real Llama executor";
        return false;
    }
    memory->committed_tokens = real_executor->context()->token_count();
    memory->kv_cache_bytes = real_executor->context()->kv_cache_bytes();
    if (error != nullptr) error->clear();
    return true;
}

std::size_t RealLlamaExecutorFactory::model_weight_payload_bytes() const {
    return impl_->model->weight_payload_bytes();
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
