// Chapter 06：声明负责 prefill、decode 和 token generation 的 CPU Generation Runtime。
// 职责：推进 Request state、写入 request cache，并把 token choice 交给 Sampler。
// 验证：Runtime smoke 和 generation 测试检查 admission、terminal handling 和 output count。
#pragma once

#include "chapter06_generation/generation_types.h"
#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/sampler.h"

#include <vector>

namespace chapter06_generation {

class Transformer;

struct PrefillChunkResult {
    std::size_t consumed_tokens = 0;
    std::size_t generated_tokens = 0;
};

struct BatchPrefillResult {
    std::size_t model_calls = 0;
    std::size_t prefilled_requests = 0;
    std::size_t consumed_tokens = 0;
};

struct BatchDecodeItem {
    Request* request = nullptr;
    const Sampler* sampler = nullptr;
};

struct BatchTokenResult {
    std::size_t batch_index = 0;
    TokenId token = 0;
    std::size_t step = 0;
};

struct BatchDecodeResult {
    std::size_t model_calls = 0;
    std::vector<BatchTokenResult> tokens;
};

class Runtime {
public:
    explicit Runtime(GenerationConfig config = {});
    Runtime(GenerationConfig config, const Transformer& transformer);

    const GenerationConfig& config() const;
    bool ready() const;
    std::size_t prefill(Request& request);
    BatchPrefillResult prefill_batch(const std::vector<Request*>& requests);
    std::size_t decode_one(Request& request, const Sampler& sampler);
    BatchDecodeResult decode_batch_one(const std::vector<BatchDecodeItem>& items);
    BatchDecodeResult commit_batch_logits(
        const std::vector<BatchDecodeItem>& items,
        const LogitsBatch& logits);
    std::size_t decode_one(Request& request, const Sampler& sampler, const LogitsProvider& logits_provider);
    std::size_t decode_one(Request& request, const Sampler& sampler, RequestModelExecutor& executor);
    PrefillChunkResult prefill_chunk(
        Request& request,
        const Sampler& sampler,
        RequestModelExecutor& executor,
        std::size_t max_prompt_tokens);
    std::size_t decode(Request& request, const Sampler& sampler);
    std::size_t decode(Request& request, const Sampler& sampler, const LogitsProvider& logits_provider);

    std::size_t generate(
        Request& request,
        const Sampler& sampler,
        const LogitsProvider& logits_provider,
        const TokenCallback& on_token = {});
    std::size_t generate(
        Request& request,
        const Sampler& sampler,
        RequestModelExecutor& executor,
        const TokenCallback& on_token = {});

    std::size_t generate(
        Request& request,
        const Sampler& sampler,
        const TokenCallback& on_token = {});

private:
    GenerationConfig config_;
    const Transformer* transformer_ = nullptr;
};

}  // namespace chapter06_generation
