// Chapter 10: collect HTTP preparation work into bounded model-specific batches.
#include "chapter10_serving/http_adapter.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace chapter10_serving {

struct CompletionRequestCollector::Impl final {
    struct Job {
        HttpRequest request;
        std::promise<CompletionPreparationResult> completion;
    };

    Impl(
        CompletionBatchPreparer prepare_batch,
        std::size_t batch_size,
        std::chrono::microseconds window)
        : preparer(std::move(prepare_batch)),
          max_batch_size(batch_size),
          collect_window(window) {
        if (!preparer || max_batch_size == 0) {
            throw std::invalid_argument(
                "completion collector requires a preparer and positive batch size");
        }
        worker = std::thread([this] { run(); });
    }

    ~Impl() {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        if (worker.joinable()) worker.join();
    }

    void run() {
        for (;;) {
            std::vector<std::unique_ptr<Job>> batch;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopping || !pending.empty(); });
                if (stopping && pending.empty()) return;
                if (pending.size() < max_batch_size && collect_window.count() > 0) {
                    changed.wait_for(lock, collect_window, [&] {
                        return stopping || pending.size() >= max_batch_size;
                    });
                }
                const std::size_t count = std::min(max_batch_size, pending.size());
                batch.reserve(count);
                for (std::size_t index = 0; index < count; ++index) {
                    batch.push_back(std::move(pending.front()));
                    pending.pop_front();
                }
            }

            std::vector<HttpRequest> requests;
            requests.reserve(batch.size());
            for (const auto& job : batch) requests.push_back(job->request);
            std::vector<CompletionPreparationResult> results;
            try {
                preparer(requests, &results);
            } catch (...) {
                results.clear();
            }
            if (results.size() != batch.size()) {
                results.assign(batch.size(), {
                    CompletionPrepareStatus::executor_unavailable,
                    {},
                    "completion batch preparer returned the wrong result count",
                });
            }
            {
                const std::lock_guard<std::mutex> lock(mutex);
                ++statistics.batch_calls;
                statistics.prepared_requests += batch.size();
                statistics.max_batch_size = std::max(
                    statistics.max_batch_size, batch.size());
            }
            for (std::size_t index = 0; index < batch.size(); ++index) {
                batch[index]->completion.set_value(std::move(results[index]));
            }
        }
    }

    CompletionBatchPreparer preparer;
    std::size_t max_batch_size = 0;
    std::chrono::microseconds collect_window{0};
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::unique_ptr<Job>> pending;
    CompletionCollectorStats statistics;
    bool stopping = false;
    std::thread worker;
};

CompletionRequestCollector::CompletionRequestCollector(
    CompletionBatchPreparer preparer,
    std::size_t max_batch_size,
    std::chrono::microseconds collect_window)
    : impl_(std::make_unique<Impl>(
          std::move(preparer), max_batch_size, collect_window)) {}

CompletionRequestCollector::~CompletionRequestCollector() = default;

CompletionPreparationResult CompletionRequestCollector::prepare(
    const HttpRequest& request) {
    auto job = std::make_unique<Impl::Job>();
    job->request = request;
    std::future<CompletionPreparationResult> result =
        job->completion.get_future();
    {
        const std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->stopping) {
            return {
                CompletionPrepareStatus::executor_unavailable,
                {},
                "completion collector is stopping",
            };
        }
        impl_->pending.push_back(std::move(job));
    }
    impl_->changed.notify_one();
    return result.get();
}

CompletionCollectorStats CompletionRequestCollector::stats() const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->statistics;
}

}  // namespace chapter10_serving
