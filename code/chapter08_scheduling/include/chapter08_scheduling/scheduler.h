// Chapter 08：声明 Round-robin 与教学 Tensor Batch 调度基线。
// 职责：选择可运行请求并调用 Chapter06 Runtime，不保存模型或 K/V 数据。
// 验证：scheduler_test.cpp 比较轮转、批量结果和流式输出归属。
#pragma once

#include "chapter06_generation/generation_types.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace chapter06_generation {
class Request;
class RequestModelExecutor;
class Runtime;
class Sampler;
}  // namespace chapter06_generation

namespace chapter08_scheduling {

using TokenStreamCallback = chapter06_generation::TokenStreamCallback;

struct ScheduledRequest {
    struct SchedulingMetadata {
        int priority = 0;
        std::size_t enqueue_sequence = 0;
        std::size_t cached_prefix_tokens = 0;
    };

    ScheduledRequest(
        std::size_t id,
        chapter06_generation::Request* request_value,
        const chapter06_generation::Sampler* sampler_value,
        std::shared_ptr<chapter06_generation::RequestModelExecutor> executor_value)
        : ScheduledRequest(
              id,
              request_value,
              sampler_value,
              std::move(executor_value),
              SchedulingMetadata{}) {}

    ScheduledRequest(
        std::size_t id,
        chapter06_generation::Request* request_value,
        const chapter06_generation::Sampler* sampler_value,
        std::shared_ptr<chapter06_generation::RequestModelExecutor> executor_value,
        SchedulingMetadata metadata)
        : request_id(id),
          request(request_value),
          sampler(sampler_value),
          executor(std::move(executor_value)),
          scheduling(metadata) {}

    std::size_t request_id = 0;
    chapter06_generation::Request* request = nullptr;
    const chapter06_generation::Sampler* sampler = nullptr;
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
    SchedulingMetadata scheduling;
};

struct SchedulerStats {
    std::size_t scheduled_requests = 0;
    std::size_t prefetched_requests = 0;
    std::size_t generated_tokens = 0;
    std::size_t completed_requests = 0;
    std::size_t model_batch_calls = 0;
    std::size_t tensor_batch_steps = 0;
};

class RoundRobinScheduler final {
public:
    SchedulerStats run_with_stats(
        chapter06_generation::Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
    std::size_t run(
        chapter06_generation::Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
};

class TensorBatchScheduler final {
public:
    SchedulerStats run_with_stats(
        chapter06_generation::Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
    std::size_t run(
        chapter06_generation::Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
};

}  // namespace chapter08_scheduling
