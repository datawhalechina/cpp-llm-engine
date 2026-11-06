// Chapter 08：声明单步计划、Continuous Batching 和执行输入交接。
// 职责：按请求与词元预算选择候选，并编排 Chapter07 缓存事务。
// 边界：不拥有网络等待队列，不保存分页 K/V payload。
#pragma once

#include "chapter08_scheduling/scheduler.h"
#include "chapter06_generation/model_executor.h"
#include "chapter07_kv_cache/block_table.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace chapter08_scheduling {

struct BatchCandidate {
    std::size_t scheduled_index = 0;
    std::size_t request_id = 0;
    chapter06_generation::Request* request = nullptr;
    bool prefill = false;
    std::size_t token_count = 0;
};

struct StepBatchPlan {
    std::vector<BatchCandidate> candidates;
    std::size_t token_budget = 0;
    std::size_t scheduled_tokens = 0;
    std::size_t prefill_requests = 0;
    std::size_t decode_requests = 0;
    std::size_t deferred_requests = 0;
};

struct PriorityAgingConfig {
    std::size_t aging_interval_steps = 1;
    std::size_t max_aging_boost = 8;
};

class PriorityAgingPolicy final {
public:
    explicit PriorityAgingPolicy(PriorityAgingConfig config = {});

    std::int64_t effective_priority(const ScheduledRequest& request) const;
    std::size_t waiting_steps(std::size_t request_id) const;
    std::size_t version() const;

    void finish_step(
        const std::vector<ScheduledRequest>& requests,
        const std::vector<std::size_t>& advanced_request_ids);
    void forget(std::size_t request_id);

private:
    PriorityAgingConfig config_;
    std::unordered_map<std::size_t, std::size_t> waiting_steps_;
    std::size_t version_ = 0;
};

class StepBatchPlanner final {
public:
    StepBatchPlan build(
        const std::vector<ScheduledRequest>& requests,
        std::size_t max_batch_tokens,
        std::size_t max_requests = 0,
        std::size_t max_prefill_chunk_tokens = 0,
        const PriorityAgingPolicy* policy = nullptr) const;
};

struct StepBatchStats {
    std::size_t scheduled_requests = 0;
    std::size_t scheduled_tokens = 0;
    std::size_t prefill_requests = 0;
    std::size_t decode_requests = 0;
    std::size_t generated_tokens = 0;
    std::size_t terminal_requests = 0;
    std::size_t deferred_requests = 0;
    std::size_t model_batch_calls = 0;
    std::size_t tensor_batch_requests = 0;
};

class ContinuousBatchScheduler final {
public:
    StepBatchStats run_step(
        chapter06_generation::Runtime& runtime,
        const std::vector<ScheduledRequest>& requests,
        std::size_t max_batch_tokens,
        std::size_t max_requests = 0,
        const TokenStreamCallback& on_token = {},
        chapter07_kv_cache::KvBlockPool* block_pool = nullptr,
        std::size_t max_prefill_chunk_tokens = 0,
        PriorityAgingPolicy* policy = nullptr,
        const chapter06_generation::ExecutorBatchRunner* batch_runner = nullptr) const;
};

struct ExecutorBatchToken {
    std::size_t scheduled_index = 0;
    std::size_t request_id = 0;
    chapter06_generation::TokenId token = 0;
    std::size_t position = 0;
    chapter07_kv_cache::BlockLocation kv_write;
};

struct ExecutorBatchInput {
    std::vector<ExecutorBatchToken> tokens;
    bool valid() const;
};

bool build_decode_executor_batch_input(
    const StepBatchPlan& plan,
    const std::vector<ScheduledRequest>& requests,
    const chapter07_kv_cache::KvBlockPool& block_pool,
    ExecutorBatchInput* output,
    std::string* error = nullptr);

struct RequestExecutionState {
    std::size_t prefill_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t token_history_tokens = 0;
    std::size_t expected_model_tokens = 0;
    chapter06_generation::ModelExecutionState model;
    bool has_block_table = false;
    std::size_t block_table_tokens = 0;

    bool token_history_consistent() const;
    bool model_cache_consistent() const;
    bool block_table_consistent() const;
    bool consistent() const;
};

RequestExecutionState inspect_request_execution_state(
    std::size_t request_id,
    const chapter06_generation::Request& request,
    const chapter06_generation::RequestModelExecutor* executor = nullptr,
    const chapter07_kv_cache::KvBlockPool* block_pool = nullptr);

}  // namespace chapter08_scheduling
