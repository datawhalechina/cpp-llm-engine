// Chapter 06：声明教学 round-robin Scheduler 和 token streaming callback。
// 职责：选择 runnable Request，不负责 model computation 或 KV Cache storage。
// 验证：Scheduler 测试检查 ordering、terminal skipping 和 per-request output accounting。
#pragma once

#include "chapter06_generation/generation_types.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace chapter06_generation {

class Request;
class Runtime;
class Sampler;
class RequestModelExecutor;

struct ScheduledRequest {
    std::size_t request_id = 0;
    Request* request = nullptr;
    const Sampler* sampler = nullptr;
    std::shared_ptr<RequestModelExecutor> executor;
};

struct SchedulerStats {
    std::size_t scheduled_requests = 0;
    std::size_t prefetched_requests = 0;
    std::size_t generated_tokens = 0;
    std::size_t completed_requests = 0;
    std::size_t model_batch_calls = 0;
    std::size_t tensor_batch_steps = 0;
};

using TokenStreamCallback = std::function<void(std::size_t request_id, TokenId token, std::size_t step)>;

class RoundRobinScheduler final {
public:
    bool ready() const;
    SchedulerStats run_with_stats(
        Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
    std::size_t run(
        Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
};

class TensorBatchScheduler final {
public:
    bool ready() const;
    SchedulerStats run_with_stats(
        Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
    std::size_t run(
        Runtime& runtime,
        std::vector<ScheduledRequest> requests,
        const TokenStreamCallback& on_token = {}) const;
};

}  // namespace chapter06_generation
