// Chapter 08：声明可校验的预计算调度计划和同步回退路径。
// 职责：为 overlap 控制面保存批次 ID 与状态快照，不拥有设备事件。
// 验证：overlap_scheduler_test.cpp 检查等价、失效和关闭回退。
#pragma once

#include "chapter08_scheduling/step_scheduler.h"

#include <cstddef>
#include <vector>

namespace chapter08_scheduling {

struct SchedulingStateSnapshot {
    std::size_t request_id = 0;
    const chapter06_generation::Request* request = nullptr;
    std::size_t prefill_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t token_history_tokens = 0;
    bool terminal = false;
    chapter06_generation::ModelExecutionState model;
    bool has_block_table = false;
    std::size_t block_table_tokens = 0;
};

struct PreparedStepBatch {
    std::size_t batch_id = 0;
    StepBatchPlan plan;
    std::vector<SchedulingStateSnapshot> snapshots;
    std::size_t max_requests = 0;
    std::size_t max_prefill_chunk_tokens = 0;
    bool has_policy = false;
    std::size_t policy_version = 0;
};

enum class OverlapDispatchStatus {
    Executed,
    Empty,
    Stale,
    SynchronousFallback,
};

struct OverlapDispatchResult {
    OverlapDispatchStatus status = OverlapDispatchStatus::Empty;
    StepBatchStats stats;
};

class OverlapScheduler final {
public:
    explicit OverlapScheduler(bool enabled = true);

    PreparedStepBatch prepare(
        const std::vector<ScheduledRequest>& requests,
        std::size_t max_batch_tokens,
        std::size_t max_requests = 0,
        const chapter07_kv_cache::KvCacheTransaction* block_pool = nullptr,
        std::size_t max_prefill_chunk_tokens = 0,
        const PriorityAgingPolicy* policy = nullptr);

    OverlapDispatchResult dispatch(
        chapter06_generation::Runtime& runtime,
        const std::vector<ScheduledRequest>& requests,
        const PreparedStepBatch& prepared,
        const TokenStreamCallback& on_token = {},
        chapter07_kv_cache::KvCacheTransaction* block_pool = nullptr,
        PriorityAgingPolicy* policy = nullptr) const;

private:
    bool enabled_ = true;
    std::size_t next_batch_id_ = 1;
};

}  // namespace chapter08_scheduling
