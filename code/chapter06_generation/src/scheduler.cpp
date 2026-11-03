// Chapter 06：实现 round-robin request progression 和 streaming callback dispatch。
// 契约：terminal、rejected request 被跳过，不触碰其 Runtime state。
// 验证：Scheduler 测试检查 fairness、completion accounting 和 request isolation。
#include "chapter06_generation/scheduler.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <utility>

namespace chapter06_generation {

bool RoundRobinScheduler::ready() const {
    return true;
}

SchedulerStats RoundRobinScheduler::run_with_stats(
    Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    SchedulerStats stats;
    stats.scheduled_requests = requests.size();

    if (!ready() || !runtime.ready()) {
        return stats;
    }

    for (auto& scheduled : requests) {
        if (scheduled.request == nullptr || scheduled.sampler == nullptr) {
            continue;
        }
        if (scheduled.executor == nullptr) {
            stats.prefetched_requests += runtime.prefill(*scheduled.request) > 0 ? 1 : 0;
        }
    }

    bool progress = true;
    while (progress) {
        progress = false;
        for (auto& scheduled : requests) {
            if (scheduled.request == nullptr || scheduled.sampler == nullptr || scheduled.request->terminal()) {
                continue;
            }

            const std::size_t before = scheduled.request->generated_size();
            const bool was_prefilled = scheduled.request->prefilled();
            const std::size_t decoded = scheduled.executor == nullptr
                ? runtime.decode_one(*scheduled.request, *scheduled.sampler)
                : runtime.decode_one(*scheduled.request, *scheduled.sampler,
                                     *scheduled.executor);
            if (decoded == 0) {
                continue;
            }

            if (!was_prefilled && scheduled.request->prefilled()) {
                ++stats.prefetched_requests;
            }

            progress = true;
            stats.generated_tokens += decoded;
            if (on_token != nullptr && scheduled.request->generated_size() > before) {
                on_token(
                    scheduled.request_id,
                    scheduled.request->generated().back(),
                    scheduled.request->generated_size() - 1);
            }
        }
    }

    for (const auto& scheduled : requests) {
        if (scheduled.request != nullptr && scheduled.request->finished()) {
            ++stats.completed_requests;
        }
    }

    return stats;
}

std::size_t RoundRobinScheduler::run(
    Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    return run_with_stats(runtime, std::move(requests), on_token).generated_tokens;
}

bool TensorBatchScheduler::ready() const {
    return true;
}

SchedulerStats TensorBatchScheduler::run_with_stats(
    Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    SchedulerStats stats;
    stats.scheduled_requests = requests.size();
    if (!ready() || !runtime.ready()) {
        return stats;
    }

    std::vector<Request*> prefill_requests;
    prefill_requests.reserve(requests.size());
    for (ScheduledRequest& scheduled : requests) {
        if (scheduled.executor == nullptr && scheduled.request != nullptr &&
            scheduled.sampler != nullptr) {
            prefill_requests.push_back(scheduled.request);
        }
    }
    const BatchPrefillResult prefill = runtime.prefill_batch(prefill_requests);
    stats.prefetched_requests = prefill.prefilled_requests;
    stats.model_batch_calls += prefill.model_calls;

    while (true) {
        std::vector<BatchDecodeItem> batch_items;
        std::vector<std::size_t> scheduled_indices;
        batch_items.reserve(requests.size());
        scheduled_indices.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            ScheduledRequest& scheduled = requests[index];
            if (scheduled.executor != nullptr || scheduled.request == nullptr ||
                scheduled.sampler == nullptr || scheduled.request->terminal()) {
                continue;
            }
            batch_items.push_back({scheduled.request, scheduled.sampler});
            scheduled_indices.push_back(index);
        }
        if (batch_items.empty()) {
            break;
        }

        const BatchDecodeResult decoded = runtime.decode_batch_one(batch_items);
        if (decoded.tokens.empty()) {
            break;
        }
        stats.model_batch_calls += decoded.model_calls;
        ++stats.tensor_batch_steps;
        stats.generated_tokens += decoded.tokens.size();
        for (const BatchTokenResult& output : decoded.tokens) {
            const ScheduledRequest& scheduled = requests[scheduled_indices[output.batch_index]];
            if (on_token != nullptr) {
                on_token(scheduled.request_id, output.token, output.step);
            }
        }
    }

    for (const ScheduledRequest& scheduled : requests) {
        if (scheduled.request != nullptr && scheduled.request->finished()) {
            ++stats.completed_requests;
        }
    }
    return stats;
}

std::size_t TensorBatchScheduler::run(
    Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    return run_with_stats(runtime, std::move(requests), on_token).generated_tokens;
}

}  // namespace chapter06_generation
