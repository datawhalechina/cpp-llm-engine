// Chapter 09：实现 C++ Service Core 的 admission、队列推进和资源释放。
// 契约：Service Core 只编排 Request；Runtime、Scheduler 和 Request 仍由 Chapter 06 提供。
// 验证：测试检查请求隔离、容量拒绝、终态处理和 cache clear。
#include "chapter09_serving/serving_core.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/serving_preview.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <utility>

namespace chapter09_serving {

namespace {

double elapsed_seconds(
    ServiceClock::time_point start,
    ServiceClock::time_point end) {
    return std::max(0.0, std::chrono::duration<double>(end - start).count());
}

}  // namespace

ServingCore::ServingCore(
    std::size_t max_inflight,
    std::size_t max_batch,
    ServiceNow now,
    std::shared_ptr<chapter06_generation::KvBlockPool> block_pool)
    : max_inflight_(max_inflight),
      max_batch_(max_batch),
      now_(std::move(now)),
      block_pool_(std::move(block_pool)) {
    if (max_inflight_ == 0 || max_batch_ == 0 || max_batch_ > max_inflight_) {
        throw std::invalid_argument("service capacity and batch size must be positive");
    }
    if (!now_) now_ = []() { return ServiceClock::now(); };
}

bool ServingCore::contains(std::size_t request_id) const {
    const auto matches = [request_id](const Entry& entry) {
        return entry.request_id == request_id;
    };
    return std::any_of(waiting_.begin(), waiting_.end(), matches) ||
           std::any_of(running_.begin(), running_.end(), matches);
}

AdmissionResult ServingCore::admit(
    std::size_t request_id,
    chapter06_generation::Request& request,
    const chapter06_generation::Sampler& sampler,
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor) {
    if (contains(request_id)) {
        ++metrics_.duplicate_total;
        return {AdmissionStatus::DuplicateId};
    }
    if (request.terminal()) {
        ++metrics_.invalid_total;
        return {AdmissionStatus::InvalidRequest};
    }
    if (inflight_size() >= max_inflight_) {
        ++metrics_.queue_full_total;
        return {AdmissionStatus::QueueFull};
    }
    waiting_.push_back({
        request_id, &request, &sampler, std::move(executor), now_(), false, false});
    ++metrics_.accepted_total;
    metrics_.peak_inflight = std::max(metrics_.peak_inflight, inflight_size());
    return {AdmissionStatus::Accepted};
}

bool ServingCore::mark_terminal(std::size_t request_id, bool timed_out) {
    const auto waiting = std::find_if(
        waiting_.begin(), waiting_.end(),
        [request_id](const Entry& entry) { return entry.request_id == request_id; });
    if (waiting != waiting_.end()) {
        if (timed_out) {
            waiting->request->mark_timed_out();
        } else {
            waiting->request->cancel();
        }
        observe_completion(*waiting, now_());
        waiting_.erase(waiting);
        if (timed_out) {
            ++metrics_.timed_out_total;
        } else {
            ++metrics_.cancelled_total;
        }
        return true;
    }

    const auto running = std::find_if(
        running_.begin(), running_.end(),
        [request_id](const Entry& entry) { return entry.request_id == request_id; });
    if (running == running_.end()) {
        return false;
    }
    if (timed_out) {
        running->request->mark_timed_out();
    } else {
        running->request->cancel();
    }
    observe_completion(*running, now_());
    if (timed_out) {
        ++metrics_.timed_out_total;
    } else {
        ++metrics_.cancelled_total;
    }
    return true;
}

bool ServingCore::cancel(std::size_t request_id) {
    return mark_terminal(request_id, false);
}

bool ServingCore::timeout(std::size_t request_id) {
    return mark_terminal(request_id, true);
}

ServiceStats ServingCore::dispatch(
    chapter06_generation::Runtime& runtime,
    const chapter06_generation::TokenStreamCallback& on_token) {
    if (block_pool_ != nullptr) {
        throw std::logic_error("block-aware ServiceCore requires dispatch_step");
    }
    ServiceStats stats;
    while (!waiting_.empty() && running_.size() < max_batch_) {
        Entry entry = std::move(waiting_.front());
        waiting_.pop_front();
        metrics_.queue_wait_seconds_sum += elapsed_seconds(entry.admitted_at, now_());
        ++metrics_.queue_wait_count;
        running_.push_back(std::move(entry));
    }
    stats.admitted = running_.size();

    std::vector<chapter06_generation::ScheduledRequest> scheduled;
    scheduled.reserve(running_.size());
    for (const Entry& entry : running_) {
        scheduled.push_back({entry.request_id, entry.request, entry.sampler, entry.executor});
    }
    const chapter06_generation::SchedulerStats scheduler_stats =
        chapter06_generation::RoundRobinScheduler{}.run_with_stats(
            runtime,
            std::move(scheduled),
            [this, &on_token](
                std::size_t request_id,
                chapter06_generation::TokenId token,
                std::size_t step) {
                Entry* entry = find_running(request_id);
                if (entry != nullptr) observe_token(*entry, now_());
                if (on_token) on_token(request_id, token, step);
            });
    stats.completed = scheduler_stats.completed_requests;
    stats.generated_tokens = scheduler_stats.generated_tokens;
    metrics_.generated_tokens_total += stats.generated_tokens;
    for (Entry& entry : running_) {
        if (entry.request != nullptr && entry.request->terminal()) {
            observe_completion(entry, now_());
        }
    }
    return stats;
}

ServiceStats ServingCore::dispatch_step(
    chapter06_generation::Runtime& runtime,
    std::size_t max_batch_tokens,
    const chapter06_generation::TokenStreamCallback& on_token) {
    ServiceStats stats;
    while (!waiting_.empty() && running_.size() < max_batch_) {
        Entry entry = std::move(waiting_.front());
        waiting_.pop_front();
        metrics_.queue_wait_seconds_sum += elapsed_seconds(entry.admitted_at, now_());
        ++metrics_.queue_wait_count;
        running_.push_back(std::move(entry));
    }
    stats.admitted = running_.size();

    std::vector<chapter06_generation::ScheduledRequest> scheduled;
    scheduled.reserve(running_.size());
    for (const Entry& entry : running_) {
        scheduled.push_back({entry.request_id, entry.request, entry.sampler, entry.executor});
    }
    const auto step_stats = chapter06_generation::ContinuousBatchScheduler{}.run_step(
        runtime,
        scheduled,
        max_batch_tokens,
        max_batch_,
        [this, &on_token](
            std::size_t request_id,
            chapter06_generation::TokenId token,
            std::size_t step) {
            Entry* entry = find_running(request_id);
            if (entry != nullptr) observe_token(*entry, now_());
            if (on_token) on_token(request_id, token, step);
        },
        block_pool_.get());
    stats.completed = static_cast<std::size_t>(std::count_if(
        running_.begin(), running_.end(),
        [](const Entry& entry) {
            return entry.request != nullptr && entry.request->finished();
        }));
    stats.generated_tokens = step_stats.generated_tokens;
    stats.scheduled_requests = step_stats.scheduled_requests;
    stats.scheduled_tokens = step_stats.scheduled_tokens;
    stats.deferred_requests = step_stats.deferred_requests;
    metrics_.generated_tokens_total += stats.generated_tokens;
    for (Entry& entry : running_) {
        if (entry.request != nullptr && entry.request->terminal()) {
            observe_completion(entry, now_());
        }
    }
    return stats;
}

ServiceStats ServingCore::dispatch_parallel_requests(
    const chapter06_generation::Runtime& runtime,
    std::size_t worker_count,
    const ParallelRequestCallback& on_request) {
    if (worker_count == 0) {
        throw std::invalid_argument("parallel request worker count must be positive");
    }
    if (block_pool_ != nullptr) {
        throw std::logic_error(
            "parallel request dispatch does not implement block-aware step execution");
    }
    while (!waiting_.empty() && running_.size() < max_batch_) {
        Entry entry = std::move(waiting_.front());
        waiting_.pop_front();
        metrics_.queue_wait_seconds_sum += elapsed_seconds(entry.admitted_at, now_());
        ++metrics_.queue_wait_count;
        running_.push_back(std::move(entry));
    }
    for (const Entry& entry : running_) {
        if (entry.request == nullptr || entry.sampler == nullptr || entry.executor == nullptr) {
            throw std::logic_error(
                "parallel request dispatch requires one executor and sampler per request");
        }
    }

    struct ParallelResult {
        std::size_t generated_tokens = 0;
        bool observed_first_token = false;
        ServiceClock::time_point first_token_at{};
        ServiceClock::time_point completion_at{};
    };

    ServiceStats stats;
    stats.admitted = running_.size();
    stats.scheduled_requests = running_.size();
    std::vector<ParallelResult> results(running_.size());
    std::atomic<std::size_t> next_index{0};
    const std::size_t active_workers = std::min(worker_count, running_.size());
    std::vector<std::thread> workers;
    workers.reserve(active_workers);
    for (std::size_t worker = 0; worker < active_workers; ++worker) {
        workers.emplace_back([&, worker]() {
            (void)worker;
            chapter06_generation::Runtime worker_runtime(runtime.config());
            while (true) {
                const std::size_t index = next_index.fetch_add(1);
                if (index >= running_.size()) return;
                Entry& entry = running_[index];
                ParallelResult& result = results[index];
                try {
                    while (!entry.request->terminal()) {
                        const std::size_t before = entry.request->generated_size();
                        const std::size_t generated = worker_runtime.decode_one(
                            *entry.request, *entry.sampler, *entry.executor);
                        if (generated == 0) {
                            if (!entry.request->terminal()) {
                                entry.request->mark_failed(
                                    "parallel request execution made no progress");
                            }
                            break;
                        }
                        result.generated_tokens += generated;
                        if (!result.observed_first_token &&
                            entry.request->generated_size() > before) {
                            result.observed_first_token = true;
                            result.first_token_at = now_();
                        }
                    }
                    result.completion_at = now_();
                } catch (...) {
                    entry.request->mark_failed("parallel request executor threw an exception");
                    result.completion_at = now_();
                }
            }
        });
    }
    for (std::thread& worker : workers) worker.join();

    for (std::size_t index = 0; index < running_.size(); ++index) {
        Entry& entry = running_[index];
        const ParallelResult& result = results[index];
        stats.generated_tokens += result.generated_tokens;
        if (result.observed_first_token) observe_token(entry, result.first_token_at);
        if (entry.request->terminal()) {
            observe_completion(entry, result.completion_at);
            if (entry.request->finished()) ++stats.completed;
        }
        if (on_request) {
            on_request({
                entry.request_id,
                result.generated_tokens,
                result.observed_first_token
                    ? elapsed_seconds(entry.admitted_at, result.first_token_at) : 0.0,
                elapsed_seconds(entry.admitted_at, result.completion_at),
            });
        }
    }
    metrics_.generated_tokens_total += stats.generated_tokens;
    return stats;
}

std::size_t ServingCore::release_completed() {
    const auto completed = [](const Entry& entry) {
        return entry.request != nullptr && entry.request->terminal();
    };
    std::size_t released = 0;
    for (Entry& entry : running_) {
        if (completed(entry)) {
            observe_completion(entry, now_());
            entry.request->cache().clear();
            if (block_pool_ != nullptr) block_pool_->release(entry.request_id);
            ++released;
            if (entry.request->finished()) {
                ++metrics_.completed_total;
            } else if (entry.request->failed()) {
                ++metrics_.failed_total;
            }
        }
    }
    running_.erase(
        std::remove_if(running_.begin(), running_.end(), completed), running_.end());
    metrics_.released_total += released;
    return released;
}

std::size_t ServingCore::waiting_size() const {
    return waiting_.size();
}

std::size_t ServingCore::running_size() const {
    return running_.size();
}

std::size_t ServingCore::inflight_size() const {
    return waiting_.size() + running_.size();
}

const ServiceMetrics& ServingCore::metrics() const {
    return metrics_;
}

const std::shared_ptr<chapter06_generation::KvBlockPool>&
ServingCore::block_pool() const {
    return block_pool_;
}

ServingCore::Entry* ServingCore::find_running(std::size_t request_id) {
    const auto entry = std::find_if(
        running_.begin(), running_.end(),
        [request_id](const Entry& candidate) {
            return candidate.request_id == request_id;
        });
    return entry == running_.end() ? nullptr : &*entry;
}

void ServingCore::observe_token(
    Entry& entry,
    ServiceClock::time_point observed_at) {
    if (!entry.first_token_observed) {
        metrics_.ttft_seconds_sum += elapsed_seconds(entry.admitted_at, observed_at);
        ++metrics_.ttft_count;
        entry.first_token_observed = true;
    }
    if (entry.request != nullptr && entry.request->terminal()) {
        observe_completion(entry, observed_at);
    }
}

void ServingCore::observe_completion(
    Entry& entry,
    ServiceClock::time_point observed_at) {
    if (entry.completion_observed) return;
    metrics_.completion_seconds_sum += elapsed_seconds(entry.admitted_at, observed_at);
    ++metrics_.completion_count;
    entry.completion_observed = true;
}

}  // namespace chapter09_serving
