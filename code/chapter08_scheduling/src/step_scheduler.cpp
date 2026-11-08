// Chapter 08：实现 StepBatchPlan、Continuous Batching 和执行输入交接。
// 契约：计划阶段只读；执行前预留，推进后提交，否则裁剪回原计数。
// 验证：step_scheduler_test.cpp 与 cache_transaction_test.cpp。
#include "chapter08_scheduling/step_scheduler.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace chapter08_scheduling {

PriorityAgingPolicy::PriorityAgingPolicy(PriorityAgingConfig config) : config_(config) {
    if (config_.aging_interval_steps == 0) {
        throw std::invalid_argument("priority aging interval must be positive");
    }
}

std::int64_t PriorityAgingPolicy::effective_priority(
    const ScheduledRequest& request) const {
    const std::size_t boost = std::min(
        waiting_steps(request.request_id) / config_.aging_interval_steps,
        config_.max_aging_boost);
    return static_cast<std::int64_t>(request.scheduling.priority) +
        static_cast<std::int64_t>(boost);
}

std::size_t PriorityAgingPolicy::waiting_steps(std::size_t request_id) const {
    const auto found = waiting_steps_.find(request_id);
    return found == waiting_steps_.end() ? 0 : found->second;
}

std::size_t PriorityAgingPolicy::version() const { return version_; }

void PriorityAgingPolicy::finish_step(
    const std::vector<ScheduledRequest>& requests,
    const std::vector<std::size_t>& advanced_request_ids) {
    const std::unordered_set<std::size_t> advanced(
        advanced_request_ids.begin(), advanced_request_ids.end());
    std::unordered_set<std::size_t> active;
    bool changed = false;
    for (const ScheduledRequest& scheduled : requests) {
        if (scheduled.request == nullptr || scheduled.request->terminal()) continue;
        active.insert(scheduled.request_id);
        std::size_t& waiting = waiting_steps_[scheduled.request_id];
        const std::size_t next = advanced.count(scheduled.request_id) > 0
            ? 0
            : waiting == std::numeric_limits<std::size_t>::max()
                ? waiting
                : waiting + 1;
        if (next != waiting) changed = true;
        waiting = next;
    }
    for (auto it = waiting_steps_.begin(); it != waiting_steps_.end();) {
        if (active.count(it->first) == 0) {
            it = waiting_steps_.erase(it);
            changed = true;
        } else {
            ++it;
        }
    }
    if (changed) ++version_;
}

void PriorityAgingPolicy::forget(std::size_t request_id) {
    if (waiting_steps_.erase(request_id) > 0) ++version_;
}

StepBatchPlan StepBatchPlanner::build(
    const std::vector<ScheduledRequest>& requests,
    std::size_t max_batch_tokens,
    std::size_t max_requests,
    std::size_t max_prefill_chunk_tokens,
    const PriorityAgingPolicy* policy) const {
    if (max_batch_tokens == 0) {
        throw std::invalid_argument("batch token budget must be positive");
    }

    StepBatchPlan plan;
    plan.token_budget = max_batch_tokens;
    const bool has_chunked_prefill = std::any_of(
        requests.begin(), requests.end(), [](const ScheduledRequest& scheduled) {
            return scheduled.request != nullptr && scheduled.executor != nullptr &&
                scheduled.sampler != nullptr && !scheduled.request->terminal() &&
                !scheduled.request->prefilled() &&
                scheduled.executor->supports_chunked_prefill();
        });
    const std::size_t reserved_prefill_tokens = has_chunked_prefill ? 1 : 0;
    const auto append_phase = [&](bool prefill) {
        const std::size_t phase_token_budget = prefill
            ? max_batch_tokens
            : max_batch_tokens - reserved_prefill_tokens;
        const std::size_t phase_request_limit =
            !prefill && has_chunked_prefill && max_requests > 0
            ? max_requests - 1
            : max_requests;
        std::vector<std::size_t> indices(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) indices[index] = index;
        std::stable_sort(
            indices.begin(), indices.end(),
            [&requests, policy, prefill](std::size_t left, std::size_t right) {
                const ScheduledRequest& left_scheduled = requests[left];
                const ScheduledRequest& right_scheduled = requests[right];
                const std::int64_t left_priority = policy == nullptr
                    ? left_scheduled.scheduling.priority
                    : policy->effective_priority(left_scheduled);
                const std::int64_t right_priority = policy == nullptr
                    ? right_scheduled.scheduling.priority
                    : policy->effective_priority(right_scheduled);
                if (left_priority != right_priority) return left_priority > right_priority;

                const std::size_t left_wait = policy == nullptr
                    ? 0
                    : policy->waiting_steps(left_scheduled.request_id);
                const std::size_t right_wait = policy == nullptr
                    ? 0
                    : policy->waiting_steps(right_scheduled.request_id);
                if (left_wait != right_wait) return left_wait > right_wait;

                if (prefill) {
                    const auto* left_request = left_scheduled.request;
                    const auto* right_request = right_scheduled.request;
                    const std::size_t left_progress = left_request == nullptr
                        ? std::numeric_limits<std::size_t>::max()
                        : left_request->prefill_offset();
                    const std::size_t right_progress = right_request == nullptr
                        ? std::numeric_limits<std::size_t>::max()
                        : right_request->prefill_offset();
                    if (left_progress != right_progress) {
                        return left_progress < right_progress;
                    }
                }
                return left_scheduled.scheduling.enqueue_sequence <
                    right_scheduled.scheduling.enqueue_sequence;
            });
        for (const std::size_t index : indices) {
            const ScheduledRequest& scheduled = requests[index];
            const bool candidate_prefill =
                scheduled.request != nullptr && !scheduled.request->prefilled();
            if (scheduled.request == nullptr || scheduled.sampler == nullptr ||
                scheduled.request->terminal() || candidate_prefill != prefill) {
                continue;
            }
            if (max_requests > 0 && plan.candidates.size() >= phase_request_limit) {
                ++plan.deferred_requests;
                continue;
            }
            if (plan.scheduled_tokens == phase_token_budget) {
                ++plan.deferred_requests;
                continue;
            }

            std::size_t token_count = 1;
            if (prefill) {
                token_count = std::max<std::size_t>(
                    1, scheduled.request->remaining_prompt_tokens());
                const bool supports_chunks = scheduled.executor != nullptr &&
                    scheduled.executor->supports_chunked_prefill();
                if (supports_chunks) {
                    token_count = std::min(
                        token_count, max_batch_tokens - plan.scheduled_tokens);
                    if (max_prefill_chunk_tokens > 0) {
                        token_count = std::min(token_count, max_prefill_chunk_tokens);
                    }
                }
            }
            if (token_count > phase_token_budget - plan.scheduled_tokens) {
                ++plan.deferred_requests;
                continue;
            }
            plan.candidates.push_back(
                {index, scheduled.request_id, scheduled.request, prefill, token_count});
            plan.scheduled_tokens += token_count;
            if (prefill) ++plan.prefill_requests;
            else ++plan.decode_requests;
        }
    };

    append_phase(false);
    append_phase(true);
    return plan;
}

StepBatchStats ContinuousBatchScheduler::run_step(
    chapter06_generation::Runtime& runtime,
    const std::vector<ScheduledRequest>& requests,
    std::size_t max_batch_tokens,
    std::size_t max_requests,
    const TokenStreamCallback& on_token,
    chapter07_kv_cache::KvCacheTransaction* block_pool,
    std::size_t max_prefill_chunk_tokens,
    PriorityAgingPolicy* policy,
    const chapter06_generation::ExecutorBatchRunner* batch_runner) const {
    const StepBatchPlan plan = StepBatchPlanner{}.build(
        requests, max_batch_tokens, max_requests, max_prefill_chunk_tokens, policy);
    StepBatchStats stats;
    stats.deferred_requests = plan.deferred_requests;
    std::vector<std::size_t> advanced_request_ids;

    struct BatchedDecode {
        const BatchCandidate* candidate = nullptr;
        const ScheduledRequest* scheduled = nullptr;
        std::size_t committed_tokens = 0;
    };
    std::vector<BatchedDecode> batched;
    std::set<std::size_t> batched_indices;
    if (batch_runner != nullptr) {
        for (const BatchCandidate& candidate : plan.candidates) {
            const ScheduledRequest& scheduled = requests[candidate.scheduled_index];
            if (candidate.prefill || scheduled.request == nullptr ||
                scheduled.executor == nullptr || scheduled.request->generated().empty()) {
                continue;
            }
            const chapter06_generation::ModelExecutionState model =
                scheduled.executor->execution_state();
            if (!model.available || model.committed_tokens == 0) continue;
            const std::size_t committed_tokens = block_pool == nullptr
                ? model.committed_tokens
                : block_pool->token_count(candidate.request_id);
            if (committed_tokens != model.committed_tokens) {
                ++stats.deferred_requests;
                batched_indices.insert(candidate.scheduled_index);
                continue;
            }
            if (block_pool != nullptr &&
                !block_pool->reserve(candidate.request_id, committed_tokens + 1)) {
                ++stats.deferred_requests;
                batched_indices.insert(candidate.scheduled_index);
                continue;
            }
            batched.push_back({&candidate, &scheduled, committed_tokens});
            batched_indices.insert(candidate.scheduled_index);
        }
    }

    if (!batched.empty()) {
        std::vector<std::shared_ptr<chapter06_generation::RequestModelExecutor>> executors;
        chapter06_generation::TokenSequence tokens;
        std::vector<std::size_t> positions;
        std::vector<std::size_t> request_ids;
        std::vector<chapter06_generation::BatchDecodeItem> runtime_items;
        executors.reserve(batched.size());
        tokens.reserve(batched.size());
        positions.reserve(batched.size());
        request_ids.reserve(batched.size());
        runtime_items.reserve(batched.size());
        for (const BatchedDecode& item : batched) {
            executors.push_back(item.scheduled->executor);
            tokens.push_back(item.scheduled->request->generated().back());
            positions.push_back(item.committed_tokens);
            request_ids.push_back(item.candidate->request_id);
            runtime_items.push_back({item.scheduled->request, item.scheduled->sampler});
            ++stats.scheduled_requests;
            ++stats.scheduled_tokens;
            ++stats.decode_requests;
        }

        std::vector<chapter06_generation::Logits> logits;
        std::string error;
        const bool model_ok = batch_runner->decode_batch(
            executors, tokens, positions, request_ids, &logits, &error);
        const chapter06_generation::BatchDecodeResult decoded = model_ok
            ? runtime.commit_batch_logits(runtime_items, logits)
            : chapter06_generation::BatchDecodeResult{};
        if (!model_ok || decoded.tokens.size() != batched.size()) {
            for (const BatchedDecode& item : batched) {
                if (block_pool != nullptr) {
                    block_pool->trim_to_tokens(
                        item.candidate->request_id, item.committed_tokens);
                }
                if (!item.scheduled->request->terminal()) {
                    item.scheduled->request->mark_failed(
                        error.empty() ? "batch executor failed" : error);
                }
                ++stats.terminal_requests;
            }
        } else {
            ++stats.model_batch_calls;
            stats.tensor_batch_requests += batched.size();
            for (const chapter06_generation::BatchTokenResult& output : decoded.tokens) {
                const BatchedDecode& item = batched[output.batch_index];
                if (block_pool != nullptr && !block_pool->commit_tokens(
                        item.candidate->request_id, item.committed_tokens + 1)) {
                    throw std::logic_error("reserved batch KV blocks cannot commit");
                }
                advanced_request_ids.push_back(item.candidate->request_id);
                ++stats.generated_tokens;
                if (on_token) {
                    on_token(
                        item.candidate->request_id, output.token, output.step);
                }
                if (item.scheduled->request->terminal()) ++stats.terminal_requests;
            }
        }
    }

    for (const BatchCandidate& candidate : plan.candidates) {
        if (batched_indices.count(candidate.scheduled_index) > 0) continue;
        const ScheduledRequest& scheduled = requests[candidate.scheduled_index];
        chapter06_generation::Request& request = *scheduled.request;
        const std::size_t before = request.generated_size();
        const std::size_t prefill_before = request.prefill_offset();
        const std::size_t committed_tokens = block_pool == nullptr
            ? 0
            : block_pool->token_count(candidate.request_id);
        if (block_pool != nullptr &&
            (candidate.token_count >
                 std::numeric_limits<std::size_t>::max() - committed_tokens ||
             !block_pool->reserve(
                 candidate.request_id, committed_tokens + candidate.token_count))) {
            ++stats.deferred_requests;
            continue;
        }

        ++stats.scheduled_requests;
        stats.scheduled_tokens += candidate.token_count;
        if (candidate.prefill) ++stats.prefill_requests;
        else ++stats.decode_requests;

        if (scheduled.executor != nullptr && candidate.prefill) {
            runtime.prefill_chunk(
                request, *scheduled.sampler, *scheduled.executor, candidate.token_count);
        } else if (scheduled.executor != nullptr) {
            runtime.decode_one(request, *scheduled.sampler, *scheduled.executor);
        } else if (candidate.prefill) {
            if (runtime.prefill(request) > 0) runtime.decode_one(request, *scheduled.sampler);
        } else {
            runtime.decode_one(request, *scheduled.sampler);
        }

        const bool advanced = request.generated_size() > before ||
            request.prefill_offset() > prefill_before;
        if (advanced) advanced_request_ids.push_back(candidate.request_id);
        if (block_pool != nullptr) {
            if (advanced) {
                if (!block_pool->commit_tokens(
                        candidate.request_id,
                        committed_tokens + candidate.token_count)) {
                    throw std::logic_error("reserved KV blocks cannot commit token count");
                }
            } else {
                block_pool->trim_to_tokens(candidate.request_id, committed_tokens);
            }
        }
        if (request.generated_size() > before) {
            ++stats.generated_tokens;
            if (on_token) {
                on_token(candidate.request_id, request.generated().back(), before);
            }
        }
        if (request.terminal()) ++stats.terminal_requests;
    }
    if (policy != nullptr) policy->finish_step(requests, advanced_request_ids);
    return stats;
}

bool ExecutorBatchInput::valid() const {
    std::set<std::size_t> request_ids;
    std::set<std::pair<std::size_t, std::size_t>> kv_slots;
    for (const ExecutorBatchToken& token : tokens) {
        if (!request_ids.insert(token.request_id).second ||
            !kv_slots.insert({token.kv_write.block, token.kv_write.slot}).second) {
            return false;
        }
    }
    return true;
}

bool build_decode_executor_batch_input(
    const StepBatchPlan& plan,
    const std::vector<ScheduledRequest>& requests,
    const chapter07_kv_cache::KvCacheTransaction& block_pool,
    ExecutorBatchInput* output,
    std::string* error) {
    if (output == nullptr) {
        if (error != nullptr) *error = "executor batch output must not be null";
        return false;
    }
    const auto fail = [&](const std::string& message) {
        if (error != nullptr) *error = message;
        return false;
    };
    ExecutorBatchInput prepared;
    for (const BatchCandidate& candidate : plan.candidates) {
        if (candidate.prefill) continue;
        if (candidate.scheduled_index >= requests.size()) {
            return fail("batch candidate index exceeds scheduled requests");
        }
        const ScheduledRequest& scheduled = requests[candidate.scheduled_index];
        if (scheduled.request == nullptr || scheduled.executor == nullptr ||
            scheduled.request->terminal() || !scheduled.request->prefilled() ||
            scheduled.request->generated().empty()) {
            return fail("decode batch candidate is not executable");
        }
        const chapter06_generation::ModelExecutionState state =
            scheduled.executor->execution_state();
        if (!state.available || !state.stores_kv_values) {
            return fail("decode batch executor does not expose KV state");
        }
        if (!block_pool.contains(candidate.request_id) ||
            block_pool.token_count(candidate.request_id) != state.committed_tokens) {
            return fail("executor position and BlockTable token count differ");
        }
        if (state.committed_tokens >= block_pool.capacity_tokens(candidate.request_id)) {
            return fail("decode batch has no reserved KV write slot");
        }
        prepared.tokens.push_back({
            candidate.scheduled_index,
            candidate.request_id,
            scheduled.request->generated().back(),
            state.committed_tokens,
            block_pool.locate_reserved(candidate.request_id, state.committed_tokens),
        });
    }
    if (!prepared.valid()) {
        return fail("decode batch contains duplicate request or KV slot");
    }
    *output = std::move(prepared);
    if (error != nullptr) error->clear();
    return true;
}

bool RequestExecutionState::token_history_consistent() const {
    return token_history_tokens == prefill_tokens + generated_tokens;
}
bool RequestExecutionState::model_cache_consistent() const {
    return !model.available || model.committed_tokens == expected_model_tokens;
}
bool RequestExecutionState::block_table_consistent() const {
    return !has_block_table || block_table_tokens == expected_model_tokens;
}
bool RequestExecutionState::consistent() const {
    return token_history_consistent() && model_cache_consistent() &&
        block_table_consistent();
}

RequestExecutionState inspect_request_execution_state(
    std::size_t request_id,
    const chapter06_generation::Request& request,
    const chapter06_generation::RequestModelExecutor* executor,
    const chapter07_kv_cache::KvCacheTransaction* block_pool) {
    RequestExecutionState state;
    state.prefill_tokens = request.prefill_offset();
    state.generated_tokens = request.generated_size();
    state.token_history_tokens = request.token_history_size();
    state.expected_model_tokens = state.prefill_tokens;
    if (state.generated_tokens > 0) {
        state.expected_model_tokens += state.generated_tokens - 1;
    }
    if (executor != nullptr) state.model = executor->execution_state();
    state.has_block_table = block_pool != nullptr && block_pool->contains(request_id);
    if (state.has_block_table) state.block_table_tokens = block_pool->token_count(request_id);
    return state;
}

}  // namespace chapter08_scheduling
