// Chapter 06：实现 token/block budget planner、单步调度和 CPU KV block pool。
// 契约：block 申请先检查完整容量；模型未推进时回滚未提交的预留。
// 验证：Chapter06 scheduler、block pool、prefill 和 transaction 专项测试。
#include "chapter06_generation/serving_preview.h"

#include "chapter06_generation/model_executor.h"
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace chapter06_generation {

StepBatchPlan StepBatchPlanner::build(
    const std::vector<ScheduledRequest>& requests,
    std::size_t max_batch_tokens,
    std::size_t max_requests,
    std::size_t max_prefill_chunk_tokens) const {
    if (max_batch_tokens == 0) {
        throw std::invalid_argument("batch token budget must be positive");
    }

    StepBatchPlan plan;
    plan.token_budget = max_batch_tokens;
    const bool has_chunked_prefill = std::any_of(
        requests.begin(), requests.end(),
        [](const ScheduledRequest& scheduled) {
            return scheduled.request != nullptr && scheduled.executor != nullptr &&
                scheduled.sampler != nullptr && !scheduled.request->terminal() &&
                !scheduled.request->prefilled() &&
                scheduled.executor->supports_chunked_prefill();
        });
    const std::size_t reserved_prefill_tokens = has_chunked_prefill ? 1 : 0;
    const auto append_phase = [&](bool prefill) {
        const std::size_t phase_token_budget = prefill
            ? max_batch_tokens : max_batch_tokens - reserved_prefill_tokens;
        const std::size_t phase_request_limit =
            !prefill && has_chunked_prefill && max_requests > 0
            ? max_requests - 1 : max_requests;
        std::vector<std::size_t> indices(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            indices[index] = index;
        }
        if (prefill) {
            std::stable_sort(
                indices.begin(), indices.end(),
                [&requests](std::size_t left, std::size_t right) {
                    const Request* left_request = requests[left].request;
                    const Request* right_request = requests[right].request;
                    const std::size_t left_progress = left_request == nullptr
                        ? std::numeric_limits<std::size_t>::max()
                        : left_request->prefill_offset();
                    const std::size_t right_progress = right_request == nullptr
                        ? std::numeric_limits<std::size_t>::max()
                        : right_request->prefill_offset();
                    return left_progress < right_progress;
                });
        }
        for (std::size_t index : indices) {
            const ScheduledRequest& scheduled = requests[index];
            const bool candidate_prefill = scheduled.request != nullptr &&
                !scheduled.request->prefilled();
            if (scheduled.request == nullptr || scheduled.sampler == nullptr ||
                scheduled.request->terminal() ||
                candidate_prefill != prefill) {
                continue;
            }
            if (max_requests > 0 &&
                plan.candidates.size() >= phase_request_limit) {
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

            plan.candidates.push_back({
                index, scheduled.request_id, scheduled.request, prefill, token_count});
            plan.scheduled_tokens += token_count;
            if (prefill) {
                ++plan.prefill_requests;
            } else {
                ++plan.decode_requests;
            }
        }
    };

    // Decode first protects active-request latency; remaining budget admits prefills.
    append_phase(false);
    append_phase(true);
    return plan;
}

StepBatchStats ContinuousBatchScheduler::run_step(
    Runtime& runtime,
    const std::vector<ScheduledRequest>& requests,
    std::size_t max_batch_tokens,
    std::size_t max_requests,
    const TokenStreamCallback& on_token,
    KvBlockPool* block_pool,
    std::size_t max_prefill_chunk_tokens) const {
    const StepBatchPlan plan = StepBatchPlanner{}.build(
        requests, max_batch_tokens, max_requests, max_prefill_chunk_tokens);
    StepBatchStats stats;
    stats.deferred_requests = plan.deferred_requests;

    for (const BatchCandidate& candidate : plan.candidates) {
        const ScheduledRequest& scheduled = requests[candidate.scheduled_index];
        Request& request = *scheduled.request;
        const std::size_t before = request.generated_size();
        const std::size_t prefill_before = request.prefill_offset();
        const std::size_t committed_tokens = block_pool == nullptr
            ? 0 : block_pool->token_count(candidate.request_id);
        if (block_pool != nullptr) {
            if (candidate.token_count >
                    std::numeric_limits<std::size_t>::max() - committed_tokens ||
                !block_pool->reserve(
                    candidate.request_id, committed_tokens + candidate.token_count)) {
                ++stats.deferred_requests;
                continue;
            }
        }

        ++stats.scheduled_requests;
        stats.scheduled_tokens += candidate.token_count;
        if (candidate.prefill) {
            ++stats.prefill_requests;
        } else {
            ++stats.decode_requests;
        }

        if (scheduled.executor != nullptr && candidate.prefill) {
            runtime.prefill_chunk(
                request,
                *scheduled.sampler,
                *scheduled.executor,
                candidate.token_count);
        } else if (scheduled.executor != nullptr) {
            runtime.decode_one(request, *scheduled.sampler, *scheduled.executor);
        } else if (candidate.prefill) {
            if (runtime.prefill(request) > 0) {
                runtime.decode_one(request, *scheduled.sampler);
            }
        } else {
            runtime.decode_one(request, *scheduled.sampler);
        }

        const bool advanced = request.generated_size() > before ||
            request.prefill_offset() > prefill_before;
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
    return stats;
}

BlockTable::BlockTable(std::size_t block_size)
    : block_size_(block_size) {
    if (block_size_ == 0) {
        throw std::invalid_argument("block size must be positive");
    }
}

std::size_t BlockTable::block_size() const {
    return block_size_;
}

std::size_t BlockTable::token_count() const {
    return token_count_;
}

std::size_t BlockTable::block_count() const {
    return physical_blocks_.size();
}

std::size_t BlockTable::capacity_tokens() const {
    return physical_blocks_.size() * block_size_;
}

const std::vector<std::size_t>& BlockTable::physical_blocks() const {
    return physical_blocks_;
}

BlockLocation BlockTable::locate(std::size_t token_index) const {
    if (token_index >= token_count_) {
        throw std::out_of_range("token index exceeds block table");
    }
    return {
        physical_blocks_[token_index / block_size_],
        token_index % block_size_,
    };
}

BlockLocation BlockTable::locate_reserved(std::size_t token_index) const {
    if (token_index >= capacity_tokens()) {
        throw std::out_of_range("token index exceeds reserved block capacity");
    }
    return {
        physical_blocks_[token_index / block_size_],
        token_index % block_size_,
    };
}

void BlockTable::append_block(std::size_t physical_block) {
    physical_blocks_.push_back(physical_block);
}

std::size_t BlockTable::remove_last_block() {
    if (physical_blocks_.empty()) {
        throw std::logic_error("cannot remove a block from an empty table");
    }
    const std::size_t block = physical_blocks_.back();
    physical_blocks_.pop_back();
    return block;
}

void BlockTable::set_token_count(std::size_t token_count) {
    if (token_count > capacity_tokens()) {
        throw std::out_of_range("token count exceeds reserved block capacity");
    }
    token_count_ = token_count;
}

KvBlockPool::KvBlockPool(std::size_t total_blocks, std::size_t block_size)
    : block_size_(block_size) {
    if (total_blocks == 0 || block_size_ == 0) {
        throw std::invalid_argument("KV block count and block size must be positive");
    }
    if (total_blocks > std::numeric_limits<std::size_t>::max() / block_size_) {
        throw std::overflow_error("KV block capacity overflows size_t");
    }
    free_.assign(total_blocks, true);
}

std::size_t KvBlockPool::total_blocks() const { return free_.size(); }

std::size_t KvBlockPool::free_blocks() const {
    return static_cast<std::size_t>(std::count(free_.begin(), free_.end(), true));
}

std::size_t KvBlockPool::used_blocks() const { return total_blocks() - free_blocks(); }

std::size_t KvBlockPool::block_size() const { return block_size_; }

bool KvBlockPool::contains(std::size_t request_id) const {
    return tables_.find(request_id) != tables_.end();
}

std::size_t KvBlockPool::block_count(std::size_t request_id) const {
    const auto table = tables_.find(request_id);
    return table == tables_.end() ? 0 : table->second.block_count();
}

std::size_t KvBlockPool::token_count(std::size_t request_id) const {
    const auto table = tables_.find(request_id);
    return table == tables_.end() ? 0 : table->second.token_count();
}

std::size_t KvBlockPool::capacity_tokens(std::size_t request_id) const {
    const auto table = tables_.find(request_id);
    return table == tables_.end() ? 0 : table->second.capacity_tokens();
}

std::vector<std::size_t> KvBlockPool::physical_blocks(std::size_t request_id) const {
    const auto table = tables_.find(request_id);
    return table == tables_.end()
        ? std::vector<std::size_t>{} : table->second.physical_blocks();
}

bool KvBlockPool::reserve(std::size_t request_id, std::size_t total_tokens) {
    const std::size_t target_blocks = required_blocks(total_tokens);
    auto table = tables_.find(request_id);
    const std::size_t current_blocks = table == tables_.end()
        ? 0 : table->second.block_count();
    if (target_blocks <= current_blocks) return true;
    const std::size_t additional_blocks = target_blocks - current_blocks;
    if (additional_blocks > free_blocks()) return false;

    if (table == tables_.end()) {
        table = tables_.emplace(request_id, BlockTable(block_size_)).first;
    }
    for (std::size_t block = 0, allocated = 0;
         block < free_.size() && allocated < additional_blocks;
         ++block) {
        if (!free_[block]) continue;
        free_[block] = false;
        table->second.append_block(block);
        ++allocated;
    }
    return true;
}

bool KvBlockPool::commit_tokens(
    std::size_t request_id,
    std::size_t total_tokens) {
    const auto table = tables_.find(request_id);
    if (table == tables_.end() ||
        total_tokens < table->second.token_count() ||
        total_tokens > table->second.capacity_tokens()) {
        return false;
    }
    table->second.set_token_count(total_tokens);
    return true;
}

bool KvBlockPool::trim_to_tokens(
    std::size_t request_id,
    std::size_t total_tokens) {
    auto table = tables_.find(request_id);
    if (table == tables_.end()) return total_tokens == 0;
    if (total_tokens > table->second.token_count()) return false;

    const std::size_t target_blocks = required_blocks(total_tokens);
    while (table->second.block_count() > target_blocks) {
        free_[table->second.remove_last_block()] = true;
    }
    table->second.set_token_count(total_tokens);
    if (target_blocks == 0) tables_.erase(table);
    return true;
}

bool KvBlockPool::release(std::size_t request_id) {
    auto table = tables_.find(request_id);
    if (table == tables_.end()) return false;
    for (std::size_t block : table->second.physical_blocks()) {
        free_[block] = true;
    }
    tables_.erase(table);
    return true;
}

BlockLocation KvBlockPool::locate(
    std::size_t request_id,
    std::size_t token_index) const {
    const auto table = tables_.find(request_id);
    if (table == tables_.end()) {
        throw std::out_of_range("request has no KV block table");
    }
    return table->second.locate(token_index);
}

BlockLocation KvBlockPool::locate_reserved(
    std::size_t request_id,
    std::size_t token_index) const {
    const auto table = tables_.find(request_id);
    if (table == tables_.end()) {
        throw std::out_of_range("request has no KV block table");
    }
    return table->second.locate_reserved(token_index);
}

std::size_t KvBlockPool::required_blocks(std::size_t token_count) const {
    return token_count / block_size_ + (token_count % block_size_ == 0 ? 0 : 1);
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
    const KvBlockPool& block_pool,
    ExecutorBatchInput* output,
    std::string* error) {
    if (output == nullptr) {
        if (error != nullptr) *error = "executor batch output must not be null";
        return false;
    }
    output->tokens.clear();
    const auto fail = [&](const std::string& message) {
        output->tokens.clear();
        if (error != nullptr) *error = message;
        return false;
    };

    for (const BatchCandidate& candidate : plan.candidates) {
        if (candidate.prefill) {
            continue;
        }
        if (candidate.scheduled_index >= requests.size()) {
            return fail("batch candidate index exceeds scheduled requests");
        }
        const ScheduledRequest& scheduled = requests[candidate.scheduled_index];
        if (scheduled.request == nullptr || scheduled.executor == nullptr ||
            scheduled.request->terminal() || !scheduled.request->prefilled() ||
            scheduled.request->generated().empty()) {
            return fail("decode batch candidate is not executable");
        }
        const ModelExecutionState state = scheduled.executor->execution_state();
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

        output->tokens.push_back({
            candidate.scheduled_index,
            candidate.request_id,
            scheduled.request->generated().back(),
            state.committed_tokens,
            block_pool.locate_reserved(candidate.request_id, state.committed_tokens),
        });
    }
    if (!output->valid()) {
        return fail("decode batch contains duplicate request or KV slot");
    }
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
    const Request& request,
    const RequestModelExecutor* executor,
    const KvBlockPool* block_pool) {
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
    if (state.has_block_table) {
        state.block_table_tokens = block_pool->token_count(request_id);
    }
    return state;
}

}  // namespace chapter06_generation
