// Chapter 06：实现 Request state transition 和 request-scoped KV Cache access。
// 契约：terminal state 保持幂等，Runtime 和 Scheduler 可以安全跳过终态请求。
// 验证：Request 和 Scheduler 测试覆盖 cancellation、timeout、rejection 和 completion。
#include "chapter06_generation/request.h"

#include <stdexcept>
#include <utility>

namespace chapter06_generation {

Request::Request(TokenSequence prompt)
    : prompt_(std::move(prompt)) {}

const TokenSequence& Request::prompt() const {
    return prompt_;
}

const TokenSequence& Request::generated() const {
    return generated_;
}

bool Request::finished() const {
    return phase_ == RequestPhase::Finished;
}

bool Request::cancelled() const {
    return phase_ == RequestPhase::Cancelled;
}

bool Request::timed_out() const {
    return phase_ == RequestPhase::TimedOut;
}

bool Request::failed() const {
    return phase_ == RequestPhase::Failed;
}

bool Request::terminal() const {
    return finished() || cancelled() || timed_out() || failed() ||
           phase_ == RequestPhase::Rejected;
}

const KVCache& Request::cache() const {
    return cache_;
}

KVCache& Request::cache() {
    return cache_;
}

const TokenSequence& Request::token_history() const {
    return cache_.tokens();
}

std::size_t Request::token_history_size() const {
    return cache_.size();
}

bool Request::prefilled() const {
    return phase_ == RequestPhase::Prefilled || phase_ == RequestPhase::Decoding ||
           phase_ == RequestPhase::Finished;
}

RequestPhase Request::phase() const {
    return phase_;
}

std::size_t Request::prompt_size() const {
    return prompt_.size();
}

std::size_t Request::prefill_offset() const {
    return prefill_offset_;
}

std::size_t Request::remaining_prompt_tokens() const {
    return prompt_.size() - prefill_offset_;
}

std::size_t Request::generated_size() const {
    return generated_.size();
}

const std::string& Request::failure_reason() const {
    return failure_reason_;
}

void Request::append_generated_token(TokenId token) {
    if (terminal()) return;
    generated_.push_back(token);
}

void Request::advance_prefill(std::size_t token_count) {
    if (terminal()) return;
    if (token_count > remaining_prompt_tokens()) {
        throw std::out_of_range("prefill progress exceeds prompt length");
    }
    prefill_offset_ += token_count;
}

void Request::mark_prefilled() {
    if (terminal()) return;
    prefill_offset_ = prompt_.size();
    phase_ = RequestPhase::Prefilled;
}

void Request::mark_decoding() {
    if (terminal()) return;
    phase_ = RequestPhase::Decoding;
}

void Request::mark_finished() {
    if (terminal()) return;
    phase_ = RequestPhase::Finished;
}

void Request::cancel() {
    if (!terminal()) phase_ = RequestPhase::Cancelled;
}

void Request::mark_timed_out() {
    if (!terminal()) phase_ = RequestPhase::TimedOut;
}

void Request::reject() {
    if (!terminal()) phase_ = RequestPhase::Rejected;
}

void Request::mark_failed(std::string reason) {
    if (!terminal()) {
        failure_reason_ = std::move(reason);
        phase_ = RequestPhase::Failed;
    }
}

}  // namespace chapter06_generation
