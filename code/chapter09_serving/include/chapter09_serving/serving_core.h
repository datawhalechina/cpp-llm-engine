// Chapter 09：声明 C++ Service Core 的 admission、队列和资源回收接口。
// 职责：管理等待/运行 Request，不重新定义 Chapter 06 Runtime 的生成语义。
// 验证：Chapter 09 测试覆盖容量、批量调度、取消、超时和完成后 cache 回收。
#pragma once

#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/scheduler.h"

#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace chapter06_generation {
class KvBlockPool;
}

namespace chapter09_serving {

using ServiceClock = std::chrono::steady_clock;
using ServiceNow = std::function<ServiceClock::time_point()>;

enum class AdmissionStatus {
    Accepted,
    QueueFull,
    DuplicateId,
    InvalidRequest,
};

struct AdmissionResult {
    AdmissionStatus status = AdmissionStatus::InvalidRequest;

    bool accepted() const { return status == AdmissionStatus::Accepted; }
};

struct ServiceStats {
    std::size_t admitted = 0;
    std::size_t completed = 0;
    std::size_t generated_tokens = 0;
    std::size_t released = 0;
    std::size_t scheduled_requests = 0;
    std::size_t scheduled_tokens = 0;
    std::size_t deferred_requests = 0;
};

struct ServiceMetrics {
    std::size_t accepted_total = 0;
    std::size_t queue_full_total = 0;
    std::size_t duplicate_total = 0;
    std::size_t invalid_total = 0;
    std::size_t cancelled_total = 0;
    std::size_t timed_out_total = 0;
    std::size_t failed_total = 0;
    std::size_t completed_total = 0;
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

struct ParallelRequestObservation {
    std::size_t request_id = 0;
    std::size_t generated_tokens = 0;
    double ttft_seconds = 0.0;
    double completion_seconds = 0.0;
};

using ParallelRequestCallback = std::function<void(const ParallelRequestObservation&)>;

class ServingCore final {
public:
    ServingCore(
        std::size_t max_inflight,
        std::size_t max_batch,
        ServiceNow now = {},
        std::shared_ptr<chapter06_generation::KvBlockPool> block_pool = {});

    AdmissionResult admit(
        std::size_t request_id,
        chapter06_generation::Request& request,
        const chapter06_generation::Sampler& sampler,
        std::shared_ptr<chapter06_generation::RequestModelExecutor> executor = {});
    bool cancel(std::size_t request_id);
    bool timeout(std::size_t request_id);

    ServiceStats dispatch(
        chapter06_generation::Runtime& runtime,
        const chapter06_generation::TokenStreamCallback& on_token = {});
    ServiceStats dispatch_step(
        chapter06_generation::Runtime& runtime,
        std::size_t max_batch_tokens,
        const chapter06_generation::TokenStreamCallback& on_token = {});
    // 每个 worker 只推进一个请求；Sampler、Executor 和自定义时钟必须支持并发访问。
    // 观测回调在全部 worker join 后由调用线程执行，不提供 token 级 streaming。
    ServiceStats dispatch_parallel_requests(
        const chapter06_generation::Runtime& runtime,
        std::size_t worker_count,
        const ParallelRequestCallback& on_request = {});
    std::size_t release_completed();

    std::size_t waiting_size() const;
    std::size_t running_size() const;
    std::size_t inflight_size() const;
    const ServiceMetrics& metrics() const;
    const std::shared_ptr<chapter06_generation::KvBlockPool>& block_pool() const;

private:
    struct Entry {
        std::size_t request_id = 0;
        chapter06_generation::Request* request = nullptr;
        const chapter06_generation::Sampler* sampler = nullptr;
        std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
        ServiceClock::time_point admitted_at;
        bool first_token_observed = false;
        bool completion_observed = false;
    };

    bool contains(std::size_t request_id) const;
    bool mark_terminal(std::size_t request_id, bool timed_out);
    Entry* find_running(std::size_t request_id);
    void observe_token(Entry& entry, ServiceClock::time_point observed_at);
    void observe_completion(Entry& entry, ServiceClock::time_point observed_at);

    std::size_t max_inflight_;
    std::size_t max_batch_;
    std::deque<Entry> waiting_;
    std::vector<Entry> running_;
    ServiceMetrics metrics_;
    ServiceNow now_;
    std::shared_ptr<chapter06_generation::KvBlockPool> block_pool_;
};

}  // namespace chapter09_serving
