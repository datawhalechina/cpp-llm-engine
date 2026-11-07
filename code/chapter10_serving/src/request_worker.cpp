// Chapter 10：实现请求级 CPU Worker 执行。
// 契约：每个任务独占 Request/Executor；worker 只共享不可变 Runtime 配置。
// 验证：Chapter10 Worker 测试和真实 TinyLlama 并行 benchmark。
#include "chapter10_serving/request_worker.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace chapter10_serving {

std::vector<WorkerResult> RequestWorkerPool::run(
    const chapter06_generation::Runtime& runtime,
    const std::vector<WorkerRequest>& requests,
    std::size_t worker_count) const {
    if (worker_count == 0) {
        throw std::invalid_argument("request worker count must be positive");
    }
    std::unordered_set<std::size_t> request_ids;
    std::unordered_set<const chapter06_generation::Request*> request_objects;
    std::unordered_set<const chapter06_generation::RequestModelExecutor*> executors;
    for (const WorkerRequest& item : requests) {
        if (item.request == nullptr || item.sampler == nullptr || item.executor == nullptr) {
            throw std::invalid_argument("request worker requires request, sampler and executor");
        }
        if (!request_ids.insert(item.request_id).second ||
            !request_objects.insert(item.request).second ||
            !executors.insert(item.executor.get()).second) {
            throw std::invalid_argument(
                "request workers require unique IDs, requests and executors");
        }
    }

    std::vector<WorkerResult> results(requests.size());
    std::atomic<std::size_t> next_index{0};
    const std::size_t active_workers = std::min(worker_count, requests.size());
    std::vector<std::thread> workers;
    workers.reserve(active_workers);
    for (std::size_t worker = 0; worker < active_workers; ++worker) {
        workers.emplace_back([&, worker]() {
            (void)worker;
            chapter06_generation::Runtime worker_runtime(runtime.config());
            while (true) {
                const std::size_t index = next_index.fetch_add(1);
                if (index >= requests.size()) return;
                const WorkerRequest& item = requests[index];
                WorkerResult& result = results[index];
                result.request_id = item.request_id;
                try {
                    while (!item.request->terminal()) {
                        const std::size_t before = item.request->generated_size();
                        const std::size_t generated = worker_runtime.decode_one(
                            *item.request, *item.sampler, *item.executor);
                        if (generated == 0) {
                            if (!item.request->terminal()) {
                                item.request->mark_failed(
                                    "request worker execution made no progress");
                            }
                            break;
                        }
                        result.generated_tokens += generated;
                        if (!result.observed_first_token &&
                            item.request->generated_size() > before) {
                            result.observed_first_token = true;
                            result.first_token_at = WorkerClock::now();
                        }
                    }
                } catch (...) {
                    item.request->mark_failed("request worker executor threw an exception");
                }
                result.completion_at = WorkerClock::now();
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    return results;
}

}  // namespace chapter10_serving
