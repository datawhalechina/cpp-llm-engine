// Chapter 06：单请求 CPU 生成编排
// 在模型执行和采样前检查上下文限制与终态，成功后提交 token 和历史。
// 验证：Runtime smoke、generation 和 streaming 测试。
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/transformer.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace chapter06_generation {

Runtime::Runtime(GenerationConfig config)
    : config_(config) {}

Runtime::Runtime(GenerationConfig config, const Transformer& transformer)
    : config_(config), transformer_(&transformer) {}

const GenerationConfig& Runtime::config() const {
    return config_;
}

bool Runtime::ready() const {
    return config_.max_new_tokens > 0;
}

std::size_t Runtime::prefill(Request& request) {
    if (!ready() || transformer_ == nullptr || request.terminal() || request.prefilled()) {
        return 0;
    }
    if (config_.max_context_tokens > 0 &&
        (request.prompt_size() > config_.max_context_tokens ||
         config_.max_new_tokens > config_.max_context_tokens - request.prompt_size())) {
        request.reject();
        return 0;
    }

    const Logits logits = transformer_->prefill(request);
    (void)logits;

    for (TokenId token : request.prompt()) {
        request.history().append(token);
    }

    request.mark_prefilled();
    return request.prompt_size();
}

BatchPrefillResult Runtime::prefill_batch(const std::vector<Request*>& requests) {
    BatchPrefillResult result;
    if (!ready() || transformer_ == nullptr) {
        return result;
    }

    std::unordered_set<Request*> unique_requests;
    for (Request* request : requests) {
        if (request != nullptr && !unique_requests.insert(request).second) {
            return result;
        }
    }

    std::vector<Request*> runnable;
    std::vector<const Request*> model_inputs;
    runnable.reserve(requests.size());
    model_inputs.reserve(requests.size());
    for (Request* request : requests) {
        if (request == nullptr || request->terminal() || request->prefilled()) {
            continue;
        }
        if (config_.max_context_tokens > 0 &&
            (request->prompt_size() > config_.max_context_tokens ||
             config_.max_new_tokens > config_.max_context_tokens - request->prompt_size())) {
            request->reject();
            continue;
        }
        runnable.push_back(request);
        model_inputs.push_back(request);
    }
    if (runnable.empty()) {
        return result;
    }

    const LogitsBatch logits = transformer_->prefill_batch(model_inputs);
    if (logits.size() != runnable.size()) {
        return result;
    }
    result.model_calls = 1;
    for (Request* request : runnable) {
        for (TokenId token : request->prompt()) {
            request->history().append(token);
        }
        request->mark_prefilled();
        ++result.prefilled_requests;
        result.consumed_tokens += request->prompt_size();
    }
    return result;
}

std::size_t Runtime::decode_one(Request& request, const Sampler& sampler) {
    if (transformer_ == nullptr) {
        return 0;
    }

    const LogitsProvider logits_provider = [this](const Request& req, std::size_t /*step*/) {
        return transformer_->decode(req);
    };

    return decode_one(request, sampler, logits_provider);
}

BatchDecodeResult Runtime::decode_batch_one(const std::vector<BatchDecodeItem>& items) {
    BatchDecodeResult result;
    if (!ready() || transformer_ == nullptr) {
        return result;
    }

    std::unordered_set<Request*> unique_requests;
    for (const BatchDecodeItem& item : items) {
        if (item.request != nullptr && !unique_requests.insert(item.request).second) {
            return result;
        }
    }

    std::vector<std::size_t> item_indices;
    std::vector<BatchDecodeItem> runnable_items;
    std::vector<const Request*> model_inputs;
    item_indices.reserve(items.size());
    runnable_items.reserve(items.size());
    model_inputs.reserve(items.size());
    for (std::size_t index = 0; index < items.size(); ++index) {
        const BatchDecodeItem& item = items[index];
        if (item.request == nullptr || item.sampler == nullptr ||
            !item.request->prefilled() || item.request->terminal()) {
            continue;
        }
        item_indices.push_back(index);
        runnable_items.push_back(item);
        model_inputs.push_back(item.request);
    }
    if (model_inputs.empty()) {
        return result;
    }

    const LogitsBatch logits = transformer_->decode_batch(model_inputs);
    if (logits.size() != model_inputs.size()) {
        return result;
    }
    result.model_calls = 1;
    auto committed = commit_batch_logits(runnable_items, logits);
    result.tokens.reserve(committed.tokens.size());
    for (auto token : committed.tokens) {
        token.batch_index = item_indices[token.batch_index];
        result.tokens.push_back(token);
    }
    return result;
}

BatchDecodeResult Runtime::commit_batch_logits(
    const std::vector<BatchDecodeItem>& items,
    const LogitsBatch& logits) {
    BatchDecodeResult result;
    if (!ready() || items.empty() || items.size() != logits.size()) return result;

    std::unordered_set<Request*> unique_requests;
    for (std::size_t index = 0; index < items.size(); ++index) {
        const BatchDecodeItem& item = items[index];
        if (item.request == nullptr || item.sampler == nullptr ||
            !item.request->prefilled() || item.request->terminal() ||
            logits[index].empty() || !unique_requests.insert(item.request).second) {
            return {};
        }
    }

    result.tokens.reserve(items.size());
    for (std::size_t index = 0; index < items.size(); ++index) {
        Request& request = *items[index].request;
        const std::size_t step = request.generated_size();
        request.mark_decoding();
        const TokenId token = items[index].sampler->sample(logits[index]);
        request.append_generated_token(token);
        request.history().append(token);
        if (token == config_.eos_token_id ||
            request.generated_size() >= config_.max_new_tokens) {
            request.mark_finished();
        }
        result.tokens.push_back({index, token, step});
    }
    return result;
}

std::size_t Runtime::decode_one(
    Request& request,
    const Sampler& sampler,
    const LogitsProvider& logits_provider) {
    if (!ready() || !request.prefilled() || request.terminal()) {
        return 0;
    }

    request.mark_decoding();
    const std::size_t step = request.generated_size();
    const Logits logits = logits_provider(request, step);
    if (logits.empty()) {
        request.mark_failed("logits provider returned empty logits");
        return 0;
    }
    const TokenId token = sampler.sample(logits);
    request.append_generated_token(token);
    request.history().append(token);
    if (token == config_.eos_token_id ||
        request.generated_size() >= config_.max_new_tokens) {
        request.mark_finished();
    }
    return 1;
}

std::size_t Runtime::decode_one(
    Request& request,
    const Sampler& sampler,
    RequestModelExecutor& executor) {
    if (!ready() || request.terminal()) return 0;
    if (!request.prefilled() && config_.max_context_tokens > 0 &&
        (request.prompt_size() > config_.max_context_tokens ||
         config_.max_new_tokens > config_.max_context_tokens - request.prompt_size())) {
        request.reject();
        return 0;
    }

    if (!request.prefilled()) {
        return prefill_chunk(
            request, sampler, executor, request.remaining_prompt_tokens()).generated_tokens;
    }

    if (request.generated().empty()) {
        request.mark_failed("model decode requires a previously generated token");
        return 0;
    }

    Logits logits;
    std::string error;
    if (!executor.decode(request.generated().back(), &logits, &error)) {
        request.mark_failed(error.empty() ? "model decode failed" : std::move(error));
        return 0;
    }
    if (logits.empty()) {
        request.mark_failed("model executor returned empty logits");
        return 0;
    }

    request.mark_decoding();
    const TokenId token = sampler.sample(logits);
    request.append_generated_token(token);
    request.history().append(token);
    if (token == config_.eos_token_id ||
        request.generated_size() >= config_.max_new_tokens) {
        request.mark_finished();
    }
    return 1;
}

PrefillChunkResult Runtime::prefill_chunk(
    Request& request,
    const Sampler& sampler,
    RequestModelExecutor& executor,
    std::size_t max_prompt_tokens) {
    PrefillChunkResult result;
    if (!ready() || request.terminal() || request.prefilled() || max_prompt_tokens == 0) {
        return result;
    }
    if (config_.max_context_tokens > 0 &&
        (request.prompt_size() > config_.max_context_tokens ||
         config_.max_new_tokens > config_.max_context_tokens - request.prompt_size())) {
        request.reject();
        return result;
    }

    const std::size_t remaining = request.remaining_prompt_tokens();
    const std::size_t chunk_size = std::min(remaining, max_prompt_tokens);
    if (chunk_size == 0) {
        request.mark_failed("model prefill requires a non-empty prompt");
        return result;
    }
    const bool starts_sequence = request.prefill_offset() == 0;
    const bool completes_prefill = chunk_size == remaining;
    if (!completes_prefill && !executor.supports_chunked_prefill()) {
        return result;
    }

    const auto first = request.prompt().begin() +
        static_cast<TokenSequence::difference_type>(request.prefill_offset());
    const TokenSequence chunk(first, first +
        static_cast<TokenSequence::difference_type>(chunk_size));
    Logits logits;
    std::string error;
    if (!executor.prefill_chunk(
            chunk,
            starts_sequence,
            completes_prefill,
            completes_prefill ? &logits : nullptr,
            &error)) {
        request.mark_failed(error.empty() ? "model prefill failed" : std::move(error));
        return result;
    }
    if (completes_prefill && logits.empty()) {
        request.mark_failed("model executor returned empty logits");
        return result;
    }

    request.advance_prefill(chunk_size);
    result.consumed_tokens = chunk_size;
    for (TokenId token : chunk) request.history().append(token);
    if (!completes_prefill) return result;

    request.mark_prefilled();
    request.mark_decoding();
    const TokenId token = sampler.sample(logits);
    request.append_generated_token(token);
    request.history().append(token);
    if (token == config_.eos_token_id ||
        request.generated_size() >= config_.max_new_tokens) {
        request.mark_finished();
    }
    result.generated_tokens = 1;
    return result;
}

std::size_t Runtime::decode(Request& request, const Sampler& sampler) {
    if (transformer_ == nullptr) {
        return 0;
    }

    const LogitsProvider logits_provider = [this](const Request& req, std::size_t /*step*/) {
        return transformer_->decode(req);
    };

    return decode(request, sampler, logits_provider);
}

std::size_t Runtime::decode(
    Request& request,
    const Sampler& sampler,
    const LogitsProvider& logits_provider) {
    if (!ready() || !request.prefilled() || request.terminal()) {
        return 0;
    }

    std::size_t generated_steps = 0;
    while (generated_steps < config_.max_new_tokens && !request.terminal()) {
        const std::size_t decoded = decode_one(request, sampler, logits_provider);
        if (decoded == 0) break;
        generated_steps += decoded;
    }

    return generated_steps;
}

std::size_t Runtime::generate(
    Request& request,
    const Sampler& sampler,
    const LogitsProvider& logits_provider,
    const TokenCallback& on_token) {
    if (!ready() || request.terminal()) {
        return 0;
    }
    if (config_.max_context_tokens > 0 &&
        (request.prompt_size() > config_.max_context_tokens ||
         config_.max_new_tokens > config_.max_context_tokens - request.prompt_size())) {
        request.reject();
        return 0;
    }
    if (!request.prefilled()) {
        for (TokenId token : request.prompt()) request.history().append(token);
        request.mark_prefilled();
    }

    std::size_t generated_steps = 0;
    while (generated_steps < config_.max_new_tokens && !request.terminal()) {
        request.mark_decoding();
        const Logits logits = logits_provider(request, generated_steps);
        if (logits.empty()) {
            request.mark_failed("logits provider returned empty logits");
            break;
        }
        const TokenId token = sampler.sample(logits);
        const std::size_t step = generated_steps;
        request.append_generated_token(token);
        request.history().append(token);
        ++generated_steps;
        if (token == config_.eos_token_id) request.mark_finished();
        if (on_token) on_token(token, step);
    }

    if (!request.terminal()) request.mark_finished();
    return generated_steps;
}

std::size_t Runtime::generate(
    Request& request,
    const Sampler& sampler,
    RequestModelExecutor& executor,
    const TokenCallback& on_token) {
    std::size_t generated_steps = 0;
    while (generated_steps < config_.max_new_tokens && !request.terminal()) {
        const std::size_t before = request.generated_size();
        const std::size_t decoded = decode_one(request, sampler, executor);
        if (decoded == 0) break;
        if (on_token && request.generated_size() > before) {
            on_token(request.generated().back(), before);
        }
        generated_steps += decoded;
    }
    return generated_steps;
}

std::size_t Runtime::generate(
    Request& request,
    const Sampler& sampler,
    const TokenCallback& on_token) {
    if (transformer_ == nullptr) {
        return 0;
    }

    prefill(request);
    std::size_t generated_steps = 0;
    while (generated_steps < config_.max_new_tokens && !request.terminal()) {
        const std::size_t before = request.generated_size();
        const std::size_t decoded = decode_one(request, sampler);
        if (decoded == 0) break;
        if (on_token && request.generated_size() > before) {
            on_token(request.generated().back(), before);
        }
        generated_steps += decoded;
    }
    return generated_steps;
}

}  // namespace chapter06_generation
