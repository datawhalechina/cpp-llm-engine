// Chapter 06：声明 Request 状态机和请求级词元历史。
// 职责：作为 Runtime、Sampler、Executor 与上层调度之间的共享交接对象。
// 验证：生命周期测试检查终态、生成词元和历史隔离。
#pragma once

#include "chapter06_generation/generation_types.h"
#include "chapter06_generation/token_history.h"

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
    const TokenHistory& history() const;
    TokenHistory& history();
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
    TokenHistory history_;
    std::string failure_reason_;
};

}  // namespace chapter06_generation
