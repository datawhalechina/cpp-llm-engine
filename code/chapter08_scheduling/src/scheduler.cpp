// Chapter 08：实现固定轮转与教学 Tensor Batch 调度基线。
// 契约：终态请求不再推进，流式事件始终携带原请求 ID。
// 验证：scheduler_test.cpp 比较顺序与批量执行的逐请求结果。
#include "chapter08_scheduling/scheduler.h"

#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <utility>

namespace chapter08_scheduling {

SchedulerStats RoundRobinScheduler::run_with_stats(
    chapter06_generation::Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    SchedulerStats stats;
    stats.scheduled_requests = requests.size();
    if (!runtime.ready()) return stats;

    for (ScheduledRequest& scheduled : requests) {
        if (scheduled.request == nullptr || scheduled.sampler == nullptr) continue;
        if (scheduled.executor == nullptr) {
            stats.prefetched_requests += runtime.prefill(*scheduled.request) > 0 ? 1 : 0;
        }
    }

    bool progress = true;
    while (progress) {
        progress = false;
        for (ScheduledRequest& scheduled : requests) {
            if (scheduled.request == nullptr || scheduled.sampler == nullptr ||
                scheduled.request->terminal()) {
                continue;
            }
            const std::size_t before = scheduled.request->generated_size();
            const bool was_prefilled = scheduled.request->prefilled();
            const std::size_t decoded = scheduled.executor == nullptr
                ? runtime.decode_one(*scheduled.request, *scheduled.sampler)
                : runtime.decode_one(
                    *scheduled.request, *scheduled.sampler, *scheduled.executor);
            if (decoded == 0) continue;
            if (!was_prefilled && scheduled.request->prefilled()) {
                ++stats.prefetched_requests;
            }
            progress = true;
            stats.generated_tokens += decoded;
            if (on_token && scheduled.request->generated_size() > before) {
                on_token(
                    scheduled.request_id,
                    scheduled.request->generated().back(),
                    scheduled.request->generated_size() - 1);
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

std::size_t RoundRobinScheduler::run(
    chapter06_generation::Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    return run_with_stats(runtime, std::move(requests), on_token).generated_tokens;
}

SchedulerStats TensorBatchScheduler::run_with_stats(
    chapter06_generation::Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    SchedulerStats stats;
    stats.scheduled_requests = requests.size();
    if (!runtime.ready()) return stats;

    std::vector<chapter06_generation::Request*> prefill_requests;
    prefill_requests.reserve(requests.size());
    for (ScheduledRequest& scheduled : requests) {
        if (scheduled.executor == nullptr && scheduled.request != nullptr &&
            scheduled.sampler != nullptr) {
            prefill_requests.push_back(scheduled.request);
        }
    }
    const chapter06_generation::BatchPrefillResult prefill =
        runtime.prefill_batch(prefill_requests);
    stats.prefetched_requests = prefill.prefilled_requests;
    stats.model_batch_calls += prefill.model_calls;

    while (true) {
        std::vector<chapter06_generation::BatchDecodeItem> batch_items;
        std::vector<std::size_t> scheduled_indices;
        for (std::size_t index = 0; index < requests.size(); ++index) {
            ScheduledRequest& scheduled = requests[index];
            if (scheduled.executor != nullptr || scheduled.request == nullptr ||
                scheduled.sampler == nullptr || scheduled.request->terminal()) {
                continue;
            }
            batch_items.push_back({scheduled.request, scheduled.sampler});
            scheduled_indices.push_back(index);
        }
        if (batch_items.empty()) break;

        const chapter06_generation::BatchDecodeResult decoded =
            runtime.decode_batch_one(batch_items);
        if (decoded.tokens.empty()) break;
        stats.model_batch_calls += decoded.model_calls;
        ++stats.tensor_batch_steps;
        stats.generated_tokens += decoded.tokens.size();
        for (const chapter06_generation::BatchTokenResult& output : decoded.tokens) {
            const ScheduledRequest& scheduled =
                requests[scheduled_indices[output.batch_index]];
            if (on_token) on_token(scheduled.request_id, output.token, output.step);
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
    chapter06_generation::Runtime& runtime,
    std::vector<ScheduledRequest> requests,
    const TokenStreamCallback& on_token) const {
    return run_with_stats(runtime, std::move(requests), on_token).generated_tokens;
}

}  // namespace chapter08_scheduling
