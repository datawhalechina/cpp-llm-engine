// Chapter 10: implement asynchronous admission, scheduling and completion delivery.
#include "chapter10_serving/async_service.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chapter10_serving {

struct AsyncRequestHandle::State final {
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<AsyncTokenEvent> events;
    AsyncRequestResult result;
    AdmissionResult admission;
    bool admission_resolved = false;
    bool done = false;
};

namespace {

void finish_state(
    const std::shared_ptr<AsyncRequestHandle::State>& state,
    AsyncRequestResult result) {
    if (state == nullptr) return;
    {
        const std::lock_guard<std::mutex> lock(state->mutex);
        if (state->done) return;
        if (!state->admission_resolved) {
            state->admission = {result.admission};
            state->admission_resolved = true;
        }
        state->result = std::move(result);
        state->done = true;
    }
    state->changed.notify_all();
}

AsyncCompletionStatus completion_status(
    const chapter06_generation::Request& request) {
    if (request.finished()) return AsyncCompletionStatus::completed;
    if (request.cancelled()) return AsyncCompletionStatus::cancelled;
    if (request.timed_out()) return AsyncCompletionStatus::timed_out;
    return AsyncCompletionStatus::failed;
}

}  // namespace

AsyncRequestHandle::AsyncRequestHandle(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

bool AsyncRequestHandle::valid() const { return state_ != nullptr; }

AdmissionResult AsyncRequestHandle::wait_admission() const {
    if (state_ == nullptr) return {AdmissionStatus::invalid_request};
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->changed.wait(lock, [&] { return state_->admission_resolved; });
    return state_->admission;
}

bool AsyncRequestHandle::wait_next(
    std::size_t* cursor,
    AsyncTokenEvent* event) const {
    if (state_ == nullptr || cursor == nullptr || event == nullptr) return false;
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->changed.wait(lock, [&] {
        return *cursor < state_->events.size() || state_->done;
    });
    if (*cursor >= state_->events.size()) return false;
    *event = state_->events[*cursor];
    ++*cursor;
    return true;
}

AsyncRequestResult AsyncRequestHandle::wait() const {
    if (state_ == nullptr) {
        return {AsyncCompletionStatus::stopped, AdmissionStatus::invalid_request, {},
                "asynchronous request handle is empty"};
    }
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->changed.wait(lock, [&] { return state_->done; });
    return state_->result;
}

struct AsyncServiceLoop::Impl final {
    struct Submission {
        std::size_t request_id = 0;
        std::unique_ptr<chapter06_generation::Request> request;
        std::shared_ptr<chapter06_generation::RequestModelExecutor> executor;
        int priority = 0;
        std::shared_ptr<AsyncRequestHandle::State> state;
    };

    struct ActiveRequest {
        std::unique_ptr<chapter06_generation::Request> request;
        std::shared_ptr<AsyncRequestHandle::State> state;
    };

    enum class ControlKind { cancel, timeout };
    struct Control {
        ControlKind kind = ControlKind::cancel;
        std::size_t request_id = 0;
    };

    Impl(
        ServiceCore& service_value,
        chapter06_generation::Runtime& runtime_value,
        const chapter06_generation::Sampler& sampler_value,
        std::size_t batch_tokens,
        std::size_t prefill_chunk_tokens,
        std::chrono::microseconds collect_window)
        : service(service_value),
          runtime(runtime_value),
          sampler(sampler_value),
          max_batch_tokens(batch_tokens),
          max_prefill_chunk_tokens(prefill_chunk_tokens),
          submission_collect_window(collect_window),
          worker([this] { run(); }) {}

    ~Impl() {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        if (worker.joinable()) worker.join();
    }

    void publish_event(
        std::size_t request_id,
        chapter06_generation::TokenId token,
        std::size_t step) {
        const auto found = active.find(request_id);
        if (found == active.end() || found->second.state == nullptr) return;
        {
            const std::lock_guard<std::mutex> lock(found->second.state->mutex);
            found->second.state->events.push_back({token, step});
        }
        found->second.state->changed.notify_all();
    }

    void update_snapshot() {
        const std::lock_guard<std::mutex> lock(mutex);
        latest.metrics = service.metrics();
        latest.dispatch = dispatch_totals;
        latest.submission_batches = submission_batches;
        latest.max_submission_batch_size = max_submission_batch_size;
        latest.waiting = service.waiting_size();
        latest.running = service.running_size();
    }

    void finish_terminal_requests() {
        std::vector<std::size_t> terminal_ids;
        for (const auto& [request_id, item] : active) {
            if (item.request != nullptr && item.request->terminal()) {
                terminal_ids.push_back(request_id);
            }
        }
        if (terminal_ids.empty()) return;
        service.release_terminal();
        update_snapshot();
        for (const std::size_t request_id : terminal_ids) {
            auto found = active.find(request_id);
            if (found == active.end()) continue;
            chapter06_generation::Request& request = *found->second.request;
            finish_state(found->second.state, {
                completion_status(request),
                AdmissionStatus::accepted,
                request.generated(),
                request.failure_reason(),
            });
            active.erase(found);
        }
    }

    void run() {
        for (;;) {
            std::deque<Submission> incoming;
            std::deque<Control> controls;
            bool should_stop = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (active.empty() && submissions.empty() && control.empty() && !stopping) {
                    changed.wait(lock, [&] {
                        return stopping || !submissions.empty() || !control.empty();
                    });
                }
                if (!stopping && active.empty() && !submissions.empty() &&
                    submission_collect_window.count() > 0) {
                    changed.wait_for(lock, submission_collect_window, [&] {
                        return stopping || !control.empty();
                    });
                }
                incoming.swap(submissions);
                controls.swap(control);
                should_stop = stopping;
            }

            if (!incoming.empty()) {
                ++submission_batches;
                max_submission_batch_size = std::max(
                    max_submission_batch_size, incoming.size());
            }

            for (Submission& item : incoming) {
                if (should_stop) {
                    finish_state(item.state, {
                        AsyncCompletionStatus::stopped,
                        AdmissionStatus::invalid_request,
                        {},
                        "asynchronous service is stopping",
                    });
                    continue;
                }
                const AdmissionResult admission = service.admit(
                    item.request_id, *item.request, sampler,
                    std::move(item.executor), item.priority);
                {
                    const std::lock_guard<std::mutex> lock(item.state->mutex);
                    item.state->admission = admission;
                    item.state->admission_resolved = true;
                }
                item.state->changed.notify_all();
                if (!admission.accepted()) {
                    finish_state(item.state, {
                        AsyncCompletionStatus::rejected, admission.status, {},
                        "request admission rejected",
                    });
                    continue;
                }
                active.emplace(item.request_id, ActiveRequest{
                    std::move(item.request), std::move(item.state)});
            }

            for (const Control& command : controls) {
                if (command.kind == ControlKind::cancel) service.cancel(command.request_id);
                else service.timeout(command.request_id);
            }
            if (should_stop) {
                for (const auto& [request_id, item] : active) {
                    (void)item;
                    service.cancel(request_id);
                }
            }

            if (!active.empty()) {
                try {
                    const DispatchStats step = service.dispatch_step(
                        runtime, max_batch_tokens,
                        [this](std::size_t request_id,
                               chapter06_generation::TokenId token,
                               std::size_t step) {
                            publish_event(request_id, token, step);
                        },
                        max_prefill_chunk_tokens);
                    dispatch_totals.promoted_requests += step.promoted_requests;
                    dispatch_totals.scheduled_requests += step.scheduled_requests;
                    dispatch_totals.scheduled_tokens += step.scheduled_tokens;
                    dispatch_totals.generated_tokens += step.generated_tokens;
                    dispatch_totals.deferred_requests += step.deferred_requests;
                    dispatch_totals.completed += step.completed;
                    dispatch_totals.model_batch_calls += step.model_batch_calls;
                    dispatch_totals.tensor_batch_requests +=
                        step.tensor_batch_requests;
                    dispatch_totals.prefill_group_calls += step.prefill_group_calls;
                    dispatch_totals.prefill_group_requests +=
                        step.prefill_group_requests;
                } catch (...) {
                    for (auto& [request_id, item] : active) {
                        (void)request_id;
                        if (item.request != nullptr && !item.request->terminal()) {
                            item.request->mark_failed(
                                "asynchronous service dispatch threw an exception");
                        }
                    }
                }
            }
            finish_terminal_requests();
            update_snapshot();
            if (should_stop && active.empty()) return;
        }
    }

    ServiceCore& service;
    chapter06_generation::Runtime& runtime;
    const chapter06_generation::Sampler& sampler;
    std::size_t max_batch_tokens = 0;
    std::size_t max_prefill_chunk_tokens = 0;
    std::chrono::microseconds submission_collect_window{0};
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::deque<Submission> submissions;
    std::deque<Control> control;
    std::unordered_map<std::size_t, ActiveRequest> active;
    AsyncServiceSnapshot latest;
    DispatchStats dispatch_totals;
    std::size_t submission_batches = 0;
    std::size_t max_submission_batch_size = 0;
    bool stopping = false;
    std::thread worker;
};

AsyncServiceLoop::AsyncServiceLoop(
    ServiceCore& service,
    chapter06_generation::Runtime& runtime,
    const chapter06_generation::Sampler& sampler,
    std::size_t max_batch_tokens,
    std::size_t max_prefill_chunk_tokens,
    std::chrono::microseconds submission_collect_window)
    : impl_(std::make_unique<Impl>(
          service, runtime, sampler, max_batch_tokens, max_prefill_chunk_tokens,
          submission_collect_window)) {}

AsyncServiceLoop::~AsyncServiceLoop() = default;

AsyncRequestHandle AsyncServiceLoop::submit(
    std::size_t request_id,
    chapter06_generation::TokenSequence prompt,
    std::shared_ptr<chapter06_generation::RequestModelExecutor> executor,
    int priority) {
    auto state = std::make_shared<AsyncRequestHandle::State>();
    {
        const std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->stopping) {
            finish_state(state, {
                AsyncCompletionStatus::stopped,
                AdmissionStatus::invalid_request,
                {},
                "asynchronous service is stopping",
            });
            return AsyncRequestHandle(std::move(state));
        }
        impl_->submissions.push_back({
            request_id,
            std::make_unique<chapter06_generation::Request>(std::move(prompt)),
            std::move(executor),
            priority,
            state,
        });
    }
    impl_->changed.notify_one();
    return AsyncRequestHandle(std::move(state));
}

void AsyncServiceLoop::cancel(std::size_t request_id) {
    {
        const std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->control.push_back({Impl::ControlKind::cancel, request_id});
    }
    impl_->changed.notify_one();
}

void AsyncServiceLoop::timeout(std::size_t request_id) {
    {
        const std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->control.push_back({Impl::ControlKind::timeout, request_id});
    }
    impl_->changed.notify_one();
}

AsyncServiceSnapshot AsyncServiceLoop::snapshot() const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->latest;
}

}  // namespace chapter10_serving
