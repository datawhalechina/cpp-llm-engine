// Chapter 10：实现通过 Chapter08 Scheduler 推进的服务状态机。
// 契约：waiting/running 共享终态释放路径，K/V 资源只通过 Chapter07 接口管理。
// 验证：Chapter10 ServiceCore 与 terminal release 测试。
#include "chapter10_serving/service_core.h"
#include "chapter10_serving/request_worker.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter07_kv_cache/block_table.h"
#include "chapter08_scheduling/step_scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace chapter10_serving {

namespace {

double elapsed_seconds(ServiceClock::time_point start, ServiceClock::time_point end) {
    return std::max(0.0, std::chrono::duration<double>(end - start).count());
}

}  // namespace

ServiceCore::ServiceCore(
    std::size_t max_inflight,
    std::size_t max_running,
    ServiceNow now,
    std::shared_ptr<chapter07_kv_cache::KvBlockPool> block_pool,
    chapter08_scheduling::PriorityAgingConfig priority_config,
    const chapter06_generation::ExecutorBatchRunner* batch_runner)
    : max_inflight_(max_inflight),
      max_running_(max_running),
      now_(std::move(now)),
      block_pool_(std::move(block_pool)),
      priority_policy_(priority_config),
      batch_runner_(batch_runner) {
    if (max_inflight_ == 0 || max_running_ == 0 || max_running_ > max_inflight_) {
        throw std::invalid_argument("service capacity must be positive and ordered");
    }
    if (!now_) now_ = [] { return ServiceClock::now(); };
}

bool ServiceCore::contains(std::size_t request_id) const {
    const auto matches = [request_id](const Entry& entry) {
        return entry.request_id == request_id;
    };
    return std::any_of(waiting_.begin(), waiting_.end(), matches) ||
           std::any_of(running_.begin(), running_.end(), matches);
}

ServiceCore::Entry* ServiceCore::find(std::size_t request_id) {
    const auto waiting = std::find_if(
        waiting_.begin(), waiting_.end(),
        [request_id](const Entry& entry) { return entry.request_id == request_id; });
    if (waiting != waiting_.end()) return &*waiting;
    const auto running = std::find_if(
        running_.begin(), running_.end(),
        [request_id](const Entry& entry) { return entry.request_id == request_id; });
    return running == running_.end() ? nullptr : &*running;
}

AdmissionResult ServiceCore::admit(
    std::size_t request_id,
    chapter06_generation::Request& request,
    const chapter06_generation::Sampler& sampler,
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor,
    int priority) {
    if (contains(request_id)) {
        ++metrics_.duplicate_total;
        return {AdmissionStatus::duplicate_id};
    }
    if (request.terminal()) {
        ++metrics_.invalid_total;
        return {AdmissionStatus::invalid_request};
    }
    if (inflight_size() >= max_inflight_) {
        ++metrics_.queue_full_total;
        return {AdmissionStatus::queue_full};
    }
    std::string bind_error;
    if (executor != nullptr && batch_runner_ != nullptr &&
        !batch_runner_->bind_request(executor, request_id, &bind_error)) {
        request.mark_failed(bind_error.empty() ? "executor request binding failed" : bind_error);
        ++metrics_.invalid_total;
        return {AdmissionStatus::invalid_request};
    }
    waiting_.push_back({
        request_id,
        &request,
        &sampler,
        std::move(executor),
        priority,
        next_enqueue_sequence_++,
        now_(),
        false,
        false});
    ++metrics_.accepted_total;
    metrics_.peak_inflight = std::max(metrics_.peak_inflight, inflight_size());
    return {AdmissionStatus::accepted};
}

bool ServiceCore::cancel(std::size_t request_id) {
    Entry* entry = find(request_id);
    if (entry == nullptr || entry->request == nullptr || entry->request->terminal()) return false;
    entry->request->cancel();
    observe_completion(*entry, now_());
    return true;
}

bool ServiceCore::timeout(std::size_t request_id) {
    Entry* entry = find(request_id);
    if (entry == nullptr || entry->request == nullptr || entry->request->terminal()) return false;
    entry->request->mark_timed_out();
    observe_completion(*entry, now_());
    return true;
}

void ServiceCore::promote_waiting() {
    while (!waiting_.empty() && running_.size() < max_running_) {
        auto next = waiting_.end();
        for (auto candidate = waiting_.begin(); candidate != waiting_.end(); ++candidate) {
            if (candidate->request == nullptr || candidate->request->terminal()) continue;
            if (next == waiting_.end() || candidate->priority > next->priority ||
                (candidate->priority == next->priority &&
                 candidate->enqueue_sequence < next->enqueue_sequence)) {
                next = candidate;
            }
        }
        if (next == waiting_.end()) return;
        Entry entry = std::move(*next);
        waiting_.erase(next);
        metrics_.queue_wait_seconds_sum += elapsed_seconds(entry.admitted_at, now_());
        ++metrics_.queue_wait_count;
        running_.push_back(std::move(entry));
    }
}

DispatchStats ServiceCore::dispatch_step(
    chapter06_generation::Runtime& runtime,
    std::size_t max_batch_tokens,
    const chapter06_generation::TokenStreamCallback& on_token,
    std::size_t max_prefill_chunk_tokens) {
    const std::size_t running_before = running_.size();
    promote_waiting();

    std::vector<chapter08_scheduling::ScheduledRequest> scheduled;
    scheduled.reserve(running_.size());
    for (const Entry& entry : running_) {
        if (entry.request != nullptr && !entry.request->terminal()) {
            scheduled.push_back({
                entry.request_id,
                entry.request,
                entry.sampler,
                entry.executor,
                {entry.priority, entry.enqueue_sequence}});
        }
    }

    const auto scheduler_stats = chapter08_scheduling::ContinuousBatchScheduler{}.run_step(
        runtime,
        scheduled,
        max_batch_tokens,
        max_running_,
        [this, &on_token](
            std::size_t request_id,
            chapter06_generation::TokenId token,
            std::size_t step) {
            Entry* entry = find(request_id);
            if (entry != nullptr) observe_token(*entry, now_());
            if (on_token) on_token(request_id, token, step);
        },
        block_pool_.get(),
        max_prefill_chunk_tokens,
        &priority_policy_,
        batch_runner_);

    for (Entry& entry : running_) {
        if (entry.request != nullptr && entry.request->terminal()) {
            observe_completion(entry, now_());
        }
    }
    metrics_.generated_tokens_total += scheduler_stats.generated_tokens;
    const std::size_t completed = static_cast<std::size_t>(std::count_if(
        running_.begin(), running_.end(), [](const Entry& entry) {
            return entry.request != nullptr && entry.request->finished();
        }));
    return {
        running_.size() - running_before,
        scheduler_stats.scheduled_requests,
        scheduler_stats.scheduled_tokens,
        scheduler_stats.generated_tokens,
        scheduler_stats.deferred_requests,
        completed,
        scheduler_stats.model_batch_calls,
        scheduler_stats.tensor_batch_requests,
    };
}

DispatchStats ServiceCore::dispatch_parallel_requests(
    const chapter06_generation::Runtime& runtime,
    std::size_t worker_count,
    const ParallelRequestCallback& on_request) {
    if (block_pool_ != nullptr) {
        throw std::logic_error(
            "request workers do not implement block-aware step execution");
    }
    const std::size_t running_before = running_.size();
    promote_waiting();

    std::vector<WorkerRequest> work;
    work.reserve(running_.size());
    for (const Entry& entry : running_) {
        if (entry.request != nullptr && !entry.request->terminal()) {
            work.push_back({
                entry.request_id, entry.request, entry.sampler, entry.executor});
        }
    }
    const auto results = RequestWorkerPool{}.run(runtime, work, worker_count);

    DispatchStats stats;
    stats.promoted_requests = running_.size() - running_before;
    stats.scheduled_requests = work.size();
    for (const WorkerResult& result : results) {
        Entry* entry = find(result.request_id);
        if (entry == nullptr || entry->request == nullptr) continue;
        stats.generated_tokens += result.generated_tokens;
        stats.scheduled_tokens += result.generated_tokens;
        if (result.observed_first_token) observe_token(*entry, result.first_token_at);
        if (entry->request->terminal()) {
            observe_completion(*entry, result.completion_at);
            if (entry->request->finished()) ++stats.completed;
        }
        if (on_request) {
            on_request({
                result.request_id,
                result.generated_tokens,
                result.observed_first_token
                    ? elapsed_seconds(entry->admitted_at, result.first_token_at) : 0.0,
                elapsed_seconds(entry->admitted_at, result.completion_at),
            });
        }
    }
    metrics_.generated_tokens_total += stats.generated_tokens;
    return stats;
}

void ServiceCore::account_terminal(const chapter06_generation::Request& request) {
    if (request.finished()) ++metrics_.completed_total;
    else if (request.cancelled()) ++metrics_.cancelled_total;
    else if (request.timed_out()) ++metrics_.timed_out_total;
    else if (request.failed()) ++metrics_.failed_total;
}

std::size_t ServiceCore::release_terminal() {
    std::size_t released = 0;
    const auto release_from = [this, &released](auto& entries) {
        entries.erase(
            std::remove_if(
                entries.begin(), entries.end(),
                [this, &released](Entry& entry) {
                    if (entry.request == nullptr || !entry.request->terminal()) return false;
                    observe_completion(entry, now_());
                    entry.request->cache().clear();
                    if (batch_runner_ != nullptr) {
                        batch_runner_->release_request(entry.request_id);
                    }
                    if (block_pool_ != nullptr) block_pool_->release(entry.request_id);
                    priority_policy_.forget(entry.request_id);
                    account_terminal(*entry.request);
                    ++released;
                    return true;
                }),
            entries.end());
    };
    release_from(waiting_);
    release_from(running_);
    metrics_.released_total += released;
    return released;
}

std::size_t ServiceCore::waiting_size() const { return waiting_.size(); }
std::size_t ServiceCore::running_size() const { return running_.size(); }
std::size_t ServiceCore::inflight_size() const {
    return waiting_.size() + running_.size();
}
const ServiceMetrics& ServiceCore::metrics() const { return metrics_; }

void ServiceCore::observe_token(Entry& entry, ServiceClock::time_point observed_at) {
    if (entry.first_token_observed) return;
    metrics_.ttft_seconds_sum += elapsed_seconds(entry.admitted_at, observed_at);
    ++metrics_.ttft_count;
    entry.first_token_observed = true;
}

void ServiceCore::observe_completion(Entry& entry, ServiceClock::time_point observed_at) {
    if (entry.completion_observed) return;
    metrics_.completion_seconds_sum += elapsed_seconds(entry.admitted_at, observed_at);
    ++metrics_.completion_count;
    entry.completion_observed = true;
}

}  // namespace chapter10_serving
