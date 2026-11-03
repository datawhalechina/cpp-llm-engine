// Chapter 09 测试支持：提供确定性 Executor、并发屏障和手动时钟。
// 职责：复用测试夹具，不包含 ServiceCore 行为断言。
// 使用范围：仅供 Chapter09 tests 目录中的组件测试包含。
#pragma once

#include "chapter06_generation/model_executor.h"
#include "chapter09_serving/serving_core.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace chapter09_serving::testing {

class FixedTokenExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    explicit FixedTokenExecutor(chapter06_generation::TokenId token) : token_(token) {}

    bool prefill(
        const chapter06_generation::TokenSequence&,
        chapter06_generation::Logits* logits,
        std::string*) override {
        ++prefill_calls;
        return write(logits);
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits* logits,
        std::string*) override {
        ++decode_calls;
        return write(logits);
    }

    std::size_t prefill_calls = 0;
    std::size_t decode_calls = 0;

private:
    bool write(chapter06_generation::Logits* logits) const {
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    chapter06_generation::TokenId token_;
};

class FailingExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    bool prefill(
        const chapter06_generation::TokenSequence&,
        chapter06_generation::Logits*,
        std::string* error) override {
        *error = "service model failure";
        return false;
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits*,
        std::string*) override {
        return false;
    }
};

class ChunkedExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    bool prefill(
        const chapter06_generation::TokenSequence& prompt,
        chapter06_generation::Logits* logits,
        std::string* error) override {
        return prefill_chunk(prompt, true, true, logits, error);
    }

    bool supports_chunked_prefill() const override { return true; }

    bool prefill_chunk(
        const chapter06_generation::TokenSequence&,
        bool,
        bool produce_logits,
        chapter06_generation::Logits* logits,
        std::string*) override {
        if (produce_logits) {
            logits->assign(8, 0.0f);
            (*logits)[3] = 1.0f;
        }
        return true;
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits* logits,
        std::string*) override {
        logits->assign(8, 0.0f);
        (*logits)[3] = 1.0f;
        return true;
    }
};

struct ParallelGate {
    std::mutex mutex;
    std::condition_variable ready;
    std::size_t arrived = 0;
    std::size_t peak = 0;
};

class CoordinatedExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    CoordinatedExecutor(
        std::shared_ptr<ParallelGate> gate,
        chapter06_generation::TokenId token)
        : gate_(std::move(gate)), token_(token) {}

    bool prefill(
        const chapter06_generation::TokenSequence&,
        chapter06_generation::Logits* logits,
        std::string* error) override {
        std::unique_lock<std::mutex> lock(gate_->mutex);
        ++gate_->arrived;
        gate_->peak = std::max(gate_->peak, gate_->arrived);
        gate_->ready.notify_all();
        if (!gate_->ready.wait_for(
                lock,
                std::chrono::seconds(2),
                [this]() { return gate_->arrived >= 2; })) {
            if (error != nullptr) *error = "parallel workers did not overlap";
            return false;
        }
        lock.unlock();
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits*,
        std::string*) override {
        return false;
    }

private:
    std::shared_ptr<ParallelGate> gate_;
    chapter06_generation::TokenId token_;
};

class ManualClock final {
public:
    ServiceClock::time_point now() const { return now_; }

    void advance(std::chrono::milliseconds duration) { now_ += duration; }

private:
    ServiceClock::time_point now_{};
};

}  // namespace chapter09_serving::testing
