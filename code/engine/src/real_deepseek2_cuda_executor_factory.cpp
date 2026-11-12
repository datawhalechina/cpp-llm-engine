// Engine CUDA：实现请求级 DeepSeek2 CUDA Runtime 适配器。
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"

#include "cpp_llm_engine/model/real_deepseek2_cuda_model.h"

#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>

namespace cpp_llm_engine {
namespace {

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
        for (const auto token : prompt) {
            const auto status = context_->forward_token(token, &current);
            if (!status.is_ok()) {
                context_->reset();
                if (error != nullptr) *error = status.message();
                return false;
            }
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
    mutable std::mutex batch_mutex;
    mutable std::unordered_map<
        std::size_t,
        std::unique_ptr<model::RealDeepSeek2CudaBatchWorkspace>>
        batch_workspaces;
};

std::unique_ptr<RealDeepSeek2CudaExecutorFactory>
RealDeepSeek2CudaExecutorFactory::load(
    const std::string& model_path,
    std::size_t context_capacity,
    std::string* error) {
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
    std::string error;
    auto context = impl_->model->create_context(impl_->context_capacity, &error);
    if (context == nullptr) return {};
    return std::make_shared<RealDeepSeek2CudaRequestExecutor>(
        std::move(context));
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
