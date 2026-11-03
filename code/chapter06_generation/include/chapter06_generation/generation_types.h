// Chapter 06：定义 Request phase、token 类型、生成限制和 logits callback。
// 契约：这些类型由 Request、Runtime、Scheduler 和 Sampler 共享。
// 验证：Chapter 06 测试检查 terminal state、context admission 和 callback 行为。
#pragma once

#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

namespace chapter06_generation {

class Request;

using TokenId = int;
using TokenSequence = std::vector<TokenId>;
using Logits = std::vector<float>;
using TokenBatch = std::vector<TokenSequence>;
using LogitsBatch = std::vector<Logits>;
using LogitsProvider = std::function<Logits(const Request&, std::size_t)>;

struct RaggedBatch {
    Logits values;
    std::vector<std::size_t> row_offsets;

    std::size_t batch_size() const {
        return row_offsets.empty() ? 0 : row_offsets.size() - 1;
    }

    bool valid() const {
        if (row_offsets.empty()) {
            return values.empty();
        }
        if (row_offsets.front() != 0 || row_offsets.back() != values.size()) {
            return false;
        }
        for (std::size_t i = 1; i < row_offsets.size(); ++i) {
            if (row_offsets[i] < row_offsets[i - 1]) {
                return false;
            }
        }
        return true;
    }
};

enum class RequestPhase {
    Initial,
    Prefilled,
    Decoding,
    Finished,
    Cancelled,
    TimedOut,
    Rejected,
    Failed,
};

inline std::string_view request_phase_name(RequestPhase phase) {
    switch (phase) {
    case RequestPhase::Initial:
        return "initial";
    case RequestPhase::Prefilled:
        return "prefilled";
    case RequestPhase::Decoding:
        return "decoding";
    case RequestPhase::Finished:
        return "finished";
    case RequestPhase::Cancelled:
        return "cancelled";
    case RequestPhase::TimedOut:
        return "timed_out";
    case RequestPhase::Rejected:
        return "rejected";
    case RequestPhase::Failed:
        return "failed";
    }

    return "unknown";
}

struct GenerationConfig {
    std::size_t max_new_tokens = 1;
    std::size_t max_context_tokens = 0;  // 0 means no tutorial admission limit.
    TokenId eos_token_id = -1;
    bool use_kv_cache = true;
};

}  // namespace chapter06_generation
