// Chapter 08 测试夹具：提供确定性和可失败的 Chapter06 模型执行器。
// 职责：共享模型行为，不包含 Scheduler 或 BlockTable 断言。
// 使用范围：Chapter08 Step、Chunked Prefill 和事务测试。
#pragma once

#include "chapter06_generation/model_executor.h"
#include "cpp_llm_engine/testing/fault_injector.h"

#include <cstddef>
#include <string>
#include <vector>

namespace chapter08_scheduling::testing {

class FixedTokenExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    explicit FixedTokenExecutor(chapter06_generation::TokenId token) : token_(token) {}

    bool prefill(
        const chapter06_generation::TokenSequence& prompt,
        chapter06_generation::Logits* logits,
        std::string*) override {
        ++prefill_calls;
        committed_tokens_ = prompt.size();
        return write(logits);
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits* logits,
        std::string*) override {
        ++decode_calls;
        ++committed_tokens_;
        return write(logits);
    }

    chapter06_generation::ModelExecutionState execution_state() const override {
        return {true, committed_tokens_, 2, true};
    }

    std::size_t prefill_calls = 0;
    std::size_t decode_calls = 0;

private:
    bool write(chapter06_generation::Logits* logits) const {
        if (logits == nullptr) return false;
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    chapter06_generation::TokenId token_;
    std::size_t committed_tokens_ = 0;
};

class ChunkedTokenExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    explicit ChunkedTokenExecutor(
        chapter06_generation::TokenId token,
        cpp_llm_engine::testing::FaultInjector* faults = nullptr)
        : token_(token), faults_(faults) {}

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
        chunk_sizes.push_back(prompt.size());
        if (faults_ != nullptr && faults_->hit("executor.prefill_chunk")) {
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

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits* logits,
        std::string*) override {
        ++committed_tokens_;
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    chapter06_generation::ModelExecutionState execution_state() const override {
        return {true, committed_tokens_, 2, true};
    }

    std::vector<std::size_t> chunk_sizes;

private:
    chapter06_generation::TokenId token_;
    cpp_llm_engine::testing::FaultInjector* faults_ = nullptr;
    std::size_t committed_tokens_ = 0;
};

}  // namespace chapter08_scheduling::testing
