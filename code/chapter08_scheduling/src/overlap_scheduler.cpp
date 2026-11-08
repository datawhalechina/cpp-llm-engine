// Chapter 08：实现预计算计划的状态版本校验与同步回退。
// 契约：请求、模型或页表变化后，旧计划只能失效，不能执行或提交。
// 验证：overlap_scheduler_test.cpp 覆盖取消、前序推进和禁用路径。
#include "chapter08_scheduling/overlap_scheduler.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"

namespace chapter08_scheduling {
namespace {

SchedulingStateSnapshot capture(
    const ScheduledRequest& scheduled,
    const chapter07_kv_cache::KvCacheTransaction* block_pool) {
    SchedulingStateSnapshot snapshot;
    snapshot.request_id = scheduled.request_id;
    snapshot.request = scheduled.request;
    if (scheduled.request != nullptr) {
        snapshot.prefill_tokens = scheduled.request->prefill_offset();
        snapshot.generated_tokens = scheduled.request->generated_size();
        snapshot.token_history_tokens = scheduled.request->token_history_size();
        snapshot.terminal = scheduled.request->terminal();
    }
    if (scheduled.executor != nullptr) {
        snapshot.model = scheduled.executor->execution_state();
    }
    snapshot.has_block_table = block_pool != nullptr &&
        block_pool->contains(scheduled.request_id);
    if (snapshot.has_block_table) {
        snapshot.block_table_tokens = block_pool->token_count(scheduled.request_id);
    }
    return snapshot;
}

bool same_state(
    const SchedulingStateSnapshot& expected,
    const ScheduledRequest& scheduled,
    const chapter07_kv_cache::KvCacheTransaction* block_pool) {
    const SchedulingStateSnapshot actual = capture(scheduled, block_pool);
    return actual.request_id == expected.request_id &&
        actual.request == expected.request &&
        actual.prefill_tokens == expected.prefill_tokens &&
        actual.generated_tokens == expected.generated_tokens &&
        actual.token_history_tokens == expected.token_history_tokens &&
        actual.terminal == expected.terminal &&
        actual.model.available == expected.model.available &&
        actual.model.committed_tokens == expected.model.committed_tokens &&
        actual.model.kv_layer_count == expected.model.kv_layer_count &&
        actual.model.stores_kv_values == expected.model.stores_kv_values &&
        actual.has_block_table == expected.has_block_table &&
        actual.block_table_tokens == expected.block_table_tokens;
}

}  // namespace

OverlapScheduler::OverlapScheduler(bool enabled) : enabled_(enabled) {}

PreparedStepBatch OverlapScheduler::prepare(
    const std::vector<ScheduledRequest>& requests,
    std::size_t max_batch_tokens,
    std::size_t max_requests,
    const chapter07_kv_cache::KvCacheTransaction* block_pool,
    std::size_t max_prefill_chunk_tokens,
    const PriorityAgingPolicy* policy) {
    PreparedStepBatch prepared;
    prepared.batch_id = next_batch_id_++;
    prepared.plan = StepBatchPlanner{}.build(
        requests, max_batch_tokens, max_requests, max_prefill_chunk_tokens, policy);
    prepared.max_requests = max_requests;
    prepared.max_prefill_chunk_tokens = max_prefill_chunk_tokens;
    prepared.has_policy = policy != nullptr;
    prepared.policy_version = policy == nullptr ? 0 : policy->version();
    prepared.snapshots.reserve(requests.size());
    for (const ScheduledRequest& scheduled : requests) {
        prepared.snapshots.push_back(capture(scheduled, block_pool));
    }
    return prepared;
}

OverlapDispatchResult OverlapScheduler::dispatch(
    chapter06_generation::Runtime& runtime,
    const std::vector<ScheduledRequest>& requests,
    const PreparedStepBatch& prepared,
    const TokenStreamCallback& on_token,
    chapter07_kv_cache::KvCacheTransaction* block_pool,
    PriorityAgingPolicy* policy) const {
    OverlapDispatchResult result;
    if (!enabled_) {
        result.status = OverlapDispatchStatus::SynchronousFallback;
        result.stats = ContinuousBatchScheduler{}.run_step(
            runtime,
            requests,
            prepared.plan.token_budget,
            prepared.max_requests,
            on_token,
            block_pool,
            prepared.max_prefill_chunk_tokens,
            policy,
            nullptr);
        return result;
    }
    if (prepared.batch_id == 0 || prepared.snapshots.size() != requests.size() ||
        prepared.has_policy != (policy != nullptr) ||
        (policy != nullptr && prepared.policy_version != policy->version())) {
        result.status = OverlapDispatchStatus::Stale;
        return result;
    }
    for (std::size_t index = 0; index < requests.size(); ++index) {
        if (!same_state(prepared.snapshots[index], requests[index], block_pool)) {
            result.status = OverlapDispatchStatus::Stale;
            return result;
        }
    }
    if (prepared.plan.candidates.empty()) {
        result.status = OverlapDispatchStatus::Empty;
        return result;
    }
    result.status = OverlapDispatchStatus::Executed;
    result.stats = ContinuousBatchScheduler{}.run_step(
        runtime,
        requests,
        prepared.plan.token_budget,
        prepared.max_requests,
        on_token,
        block_pool,
        prepared.max_prefill_chunk_tokens,
        policy,
        nullptr);
    return result;
}

}  // namespace chapter08_scheduling
