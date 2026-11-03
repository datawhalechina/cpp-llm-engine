// Chapter 06 测试夹具：提供确定性 token 执行器和可失败的分块 Prefill 执行器。
// 职责：共享模型行为，不包含 Scheduler、BlockTable 或 Request 断言。
// 使用范围：Chapter06 多请求调度专项测试。
#pragma once

#include "chapter06_generation/model_executor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace chapter06_generation::testing {

class FixedTokenExecutor final : public RequestModelExecutor {
public:
    explicit FixedTokenExecutor(TokenId token) : token_(token) {}

    bool prefill(const TokenSequence& prompt, Logits* logits, std::string*) override {
        ++prefill_calls;
        committed_tokens_ = prompt.size();
        return write(logits);
    }

    bool decode(TokenId, Logits* logits, std::string*) override {
        ++decode_calls;
        ++committed_tokens_;
        return write(logits);
    }

    ModelExecutionState execution_state() const override {
        return {true, committed_tokens_, 2, true};
    }

    std::size_t prefill_calls = 0;
    std::size_t decode_calls = 0;

private:
    bool write(Logits* logits) const {
        if (logits == nullptr) return false;
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    TokenId token_;
    std::size_t committed_tokens_ = 0;
};

class ChunkedTokenExecutor final : public RequestModelExecutor {
public:
    explicit ChunkedTokenExecutor(TokenId token, bool fail_first_chunk = false)
        : token_(token), fail_first_chunk_(fail_first_chunk) {}

    bool prefill(
        const TokenSequence& prompt,
        Logits* logits,
        std::string* error) override {
        return prefill_chunk(prompt, true, true, logits, error);
    }

    bool supports_chunked_prefill() const override { return true; }

    bool prefill_chunk(
        const TokenSequence& prompt,
        bool start_new_sequence,
        bool produce_logits,
        Logits* logits,
        std::string* error) override {
        chunk_sizes.push_back(prompt.size());
        starts.push_back(start_new_sequence);
        finals.push_back(produce_logits);
        if (fail_first_chunk_ && chunk_sizes.size() == 1) {
            if (error != nullptr) *error = "synthetic chunk failure";
            return false;
        }
        if ((!produce_logits && logits != nullptr) || (produce_logits && logits == nullptr)) {
            if (error != nullptr) *error = "logits output does not match chunk phase";
            return false;
        }
        if (start_new_sequence) committed_tokens_ = 0;
        committed_tokens_ += prompt.size();
        if (produce_logits) {
            logits->assign(8, 0.0f);
            (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        }
        return true;
    }

    bool decode(TokenId, Logits* logits, std::string*) override {
        ++committed_tokens_;
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    ModelExecutionState execution_state() const override {
        return {true, committed_tokens_, 2, true};
    }

    std::vector<std::size_t> chunk_sizes;
    std::vector<bool> starts;
    std::vector<bool> finals;

private:
    TokenId token_;
    bool fail_first_chunk_ = false;
    std::size_t committed_tokens_ = 0;
};

}  // namespace chapter06_generation::testing
