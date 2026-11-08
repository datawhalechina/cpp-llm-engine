// Chapter 06：实现 Request 状态转换和请求级词元历史访问。
// 契约：终态保持幂等，Runtime 和上层调度可以安全跳过终态请求。
// 验证入口：chapter06_request_contract_test。
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

const TokenHistory& Request::history() const {
    return history_;
}

TokenHistory& Request::history() {
    return history_;
}

const TokenSequence& Request::token_history() const {
    return history_.tokens();
}

std::size_t Request::token_history_size() const {
    return history_.size();
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

bool Request::adopt_prefilled_prefix(std::size_t token_count) {
    if (terminal() || phase_ != RequestPhase::Initial || prefill_offset_ != 0 ||
        !history_.empty() || token_count == 0 || token_count >= prompt_.size()) {
        return false;
    }
    for (std::size_t index = 0; index < token_count; ++index) {
        history_.append(prompt_[index]);
    }
    prefill_offset_ = token_count;
    return true;
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
