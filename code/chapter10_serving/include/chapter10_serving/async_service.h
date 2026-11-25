// Chapter 10: declare the single-owner asynchronous service loop.
// The loop is the only thread allowed to mutate ServiceCore.
#pragma once

#include "chapter06_generation/generation_types.h"
#include "chapter06_generation/model_executor.h"
#include "chapter10_serving/service_core.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

namespace chapter06_generation {
class Runtime;
class Sampler;
}  // namespace chapter06_generation

namespace chapter10_serving {

struct AsyncTokenEvent {
    chapter06_generation::TokenId token = 0;
    std::size_t step = 0;
};

enum class AsyncCompletionStatus {
    completed,
    cancelled,
    timed_out,
    failed,
    rejected,
    stopped,
};

struct AsyncRequestResult {
    AsyncCompletionStatus status = AsyncCompletionStatus::failed;
    AdmissionStatus admission = AdmissionStatus::invalid_request;
    chapter06_generation::TokenSequence generated;
    std::string error;
};

class AsyncRequestHandle final {
public:
    struct State;

    AsyncRequestHandle() = default;

    bool valid() const;
    AdmissionResult wait_admission() const;
    bool wait_next(std::size_t* cursor, AsyncTokenEvent* event) const;
    AsyncRequestResult wait() const;

private:
    explicit AsyncRequestHandle(std::shared_ptr<State> state);

    std::shared_ptr<State> state_;
    friend class AsyncServiceLoop;
};

struct AsyncServiceSnapshot {
    ServiceMetrics metrics;
    DispatchStats dispatch;
    std::size_t submission_batches = 0;
    std::size_t max_submission_batch_size = 0;
    std::size_t waiting = 0;
    std::size_t running = 0;
};

class AsyncServiceLoop final {
public:
    AsyncServiceLoop(
        ServiceCore& service,
        chapter06_generation::Runtime& runtime,
        const chapter06_generation::Sampler& sampler,
        std::size_t max_batch_tokens = 1024,
        std::size_t max_prefill_chunk_tokens = 0,
        std::chrono::microseconds submission_collect_window =
            std::chrono::microseconds(0));
    ~AsyncServiceLoop();

    AsyncServiceLoop(const AsyncServiceLoop&) = delete;
    AsyncServiceLoop& operator=(const AsyncServiceLoop&) = delete;

    AsyncRequestHandle submit(
        std::size_t request_id,
        chapter06_generation::TokenSequence prompt,
        std::shared_ptr<chapter06_generation::RequestModelExecutor> executor = {},
        int priority = 0);
    void cancel(std::size_t request_id);
    void timeout(std::size_t request_id);
    AsyncServiceSnapshot snapshot() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace chapter10_serving
