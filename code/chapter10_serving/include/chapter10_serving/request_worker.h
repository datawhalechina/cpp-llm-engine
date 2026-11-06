// Chapter 10：声明请求级 CPU Worker 执行契约。
// 职责：并行推进彼此隔离的 Request/Executor，不负责准入、调度策略或资源回收。
// 验证：request_worker_test 检查执行重叠、输出隔离和异常终态。
#pragma once

#include "chapter06_generation/model_executor.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

namespace chapter06_generation {
class Request;
class Runtime;
class Sampler;
}  // namespace chapter06_generation

namespace chapter10_serving {

using WorkerClock = std::chrono::steady_clock;

struct WorkerRequest {
    std::size_t request_id = 0;
    chapter06_generation::Request* request = nullptr;
    const chapter06_generation::Sampler* sampler = nullptr;
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
};

struct WorkerResult {
    std::size_t request_id = 0;
    std::size_t generated_tokens = 0;
    bool observed_first_token = false;
    WorkerClock::time_point first_token_at{};
    WorkerClock::time_point completion_at{};
};

class RequestWorkerPool final {
public:
    std::vector<WorkerResult> run(
        const chapter06_generation::Runtime& runtime,
        const std::vector<WorkerRequest>& requests,
        std::size_t worker_count) const;
};

}  // namespace chapter10_serving
