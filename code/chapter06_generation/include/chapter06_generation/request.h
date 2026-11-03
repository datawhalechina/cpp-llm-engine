// Chapter 06：声明 Request state machine 和 request-scoped KV Cache ownership。
// 职责：作为 Runtime、Scheduler、Sampler 和 streaming 之间的共享交接对象。
// 验证：生命周期测试检查 terminal transition、generated token 和 cache isolation。
#pragma once

#include "chapter06_generation/generation_types.h"
#include "chapter06_generation/kv_cache.h"

#include <string>

namespace chapter06_generation {

class Request {
public:
    Request() = default;
    explicit Request(TokenSequence prompt);

    const TokenSequence& prompt() const;
    const TokenSequence& generated() const;
    bool finished() const;
    bool cancelled() const;
    bool timed_out() const;
    bool failed() const;
    bool terminal() const;
    const KVCache& cache() const;
    KVCache& cache();
    const TokenSequence& token_history() const;
    std::size_t token_history_size() const;
    bool prefilled() const;
    RequestPhase phase() const;
    std::size_t prompt_size() const;
    std::size_t prefill_offset() const;
    std::size_t remaining_prompt_tokens() const;
    std::size_t generated_size() const;
    const std::string& failure_reason() const;

    void append_generated_token(TokenId token);
    void advance_prefill(std::size_t token_count);
    void mark_prefilled();
    void mark_decoding();
    void mark_finished();
    void cancel();
    void mark_timed_out();
    void reject();
    void mark_failed(std::string reason);

private:
    TokenSequence prompt_;
    TokenSequence generated_;
    std::size_t prefill_offset_ = 0;
    RequestPhase phase_ = RequestPhase::Initial;
    KVCache cache_;
    std::string failure_reason_;
};

}  // namespace chapter06_generation
