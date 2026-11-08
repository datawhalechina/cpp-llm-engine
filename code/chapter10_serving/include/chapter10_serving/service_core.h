// Chapter 10：声明服务准入、等待/运行状态和统一终态释放接口。
// 职责：把请求提交给 Chapter08 Scheduler，不重新实现生成或 K/V 策略。
// 线程约束：对象本身不加锁，调用方必须串行执行准入、调度、终态和指标操作。
// 验证：Chapter10 lifecycle、cache 和 HTTP contract 测试。
#pragma once

#include "chapter06_generation/generation_types.h"
#include "chapter06_generation/model_executor.h"
#include "chapter08_scheduling/step_scheduler.h"

#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace chapter06_generation {
class Request;
class Runtime;
class Sampler;
}  // namespace chapter06_generation

namespace chapter07_kv_cache {
class KvCacheTransaction;
}

namespace chapter10_serving {

using ServiceClock = std::chrono::steady_clock;
using ServiceNow = std::function<ServiceClock::time_point()>;

enum class AdmissionStatus {
    accepted,
    queue_full,
    duplicate_id,
    invalid_request,
};

struct AdmissionResult {
    AdmissionStatus status = AdmissionStatus::invalid_request;
    bool accepted() const { return status == AdmissionStatus::accepted; }
};

struct DispatchStats {
    std::size_t promoted_requests = 0;
    std::size_t scheduled_requests = 0;
    std::size_t scheduled_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t deferred_requests = 0;
    std::size_t completed = 0;
    std::size_t model_batch_calls = 0;
    std::size_t tensor_batch_requests = 0;
};

struct ParallelRequestObservation {
    std::size_t request_id = 0;
    std::size_t generated_tokens = 0;
    double ttft_seconds = 0.0;
    double completion_seconds = 0.0;
};

using ParallelRequestCallback = std::function<void(const ParallelRequestObservation&)>;

struct ServiceMetrics {
    std::size_t accepted_total = 0;
    std::size_t queue_full_total = 0;
    std::size_t duplicate_total = 0;
    std::size_t invalid_total = 0;
    std::size_t completed_total = 0;
    std::size_t cancelled_total = 0;
    std::size_t timed_out_total = 0;
    std::size_t failed_total = 0;
    std::size_t released_total = 0;
    std::size_t generated_tokens_total = 0;
    std::size_t peak_inflight = 0;
    double queue_wait_seconds_sum = 0.0;
    std::size_t queue_wait_count = 0;
    double ttft_seconds_sum = 0.0;
    std::size_t ttft_count = 0;
    double completion_seconds_sum = 0.0;
    std::size_t completion_count = 0;
};

class ServiceCore final {
public:
    ServiceCore(
        std::size_t max_inflight,
        std::size_t max_running,
        ServiceNow now = {},
        std::shared_ptr<chapter07_kv_cache::KvCacheTransaction> block_pool = {},
        chapter08_scheduling::PriorityAgingConfig priority_config = {},
        const chapter06_generation::ExecutorBatchRunner* batch_runner = nullptr);

    AdmissionResult admit(
        std::size_t request_id,
        chapter06_generation::Request& request,
        const chapter06_generation::Sampler& sampler,
        std::shared_ptr<chapter06_generation::RequestModelExecutor> executor = {},
        int priority = 0);
    bool cancel(std::size_t request_id);
    bool timeout(std::size_t request_id);

    DispatchStats dispatch_step(
        chapter06_generation::Runtime& runtime,
        std::size_t max_batch_tokens,
        const chapter06_generation::TokenStreamCallback& on_token = {},
        std::size_t max_prefill_chunk_tokens = 0);
    DispatchStats dispatch_parallel_requests(
        const chapter06_generation::Runtime& runtime,
        std::size_t worker_count,
        const ParallelRequestCallback& on_request = {});
    std::size_t release_terminal();

    std::size_t waiting_size() const;
    std::size_t running_size() const;
    std::size_t inflight_size() const;
    const ServiceMetrics& metrics() const;

private:
    struct Entry {
        std::size_t request_id = 0;
        chapter06_generation::Request* request = nullptr;
        const chapter06_generation::Sampler* sampler = nullptr;
        std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
        int priority = 0;
        std::size_t enqueue_sequence = 0;
        ServiceClock::time_point admitted_at;
        bool first_token_observed = false;
        bool completion_observed = false;
    };

    bool contains(std::size_t request_id) const;
    Entry* find(std::size_t request_id);
    void observe_token(Entry& entry, ServiceClock::time_point observed_at);
    void observe_completion(Entry& entry, ServiceClock::time_point observed_at);
    void promote_waiting();
    void account_terminal(const chapter06_generation::Request& request);

    std::size_t max_inflight_;
    std::size_t max_running_;
    std::deque<Entry> waiting_;
    std::vector<Entry> running_;
    ServiceNow now_;
    std::shared_ptr<chapter07_kv_cache::KvCacheTransaction> block_pool_;
    chapter08_scheduling::PriorityAgingPolicy priority_policy_;
    const chapter06_generation::ExecutorBatchRunner* batch_runner_ = nullptr;
    std::size_t next_enqueue_sequence_ = 0;
    ServiceMetrics metrics_;
};

}  // namespace chapter10_serving
