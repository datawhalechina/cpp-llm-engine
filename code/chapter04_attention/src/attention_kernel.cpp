// Chapter 04：CPU Attention 参考实现
// 按 head 计算 QK 分数、因果遮罩、稳定 Softmax 和 Value 加权和。
// query_position_offset 表示第一个 query token 在完整序列中的位置。
// 验证：chapter04_attention_test.cpp。
#include "chapter04_attention/attention_kernel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace chapter04_attention {

cpp_llm_engine::core::Status cpu_reference_attention_kernel(
    const cpp_llm_engine::core::AttentionRequest& request) {
    if (request.query == nullptr || request.key == nullptr || request.value == nullptr ||
        request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("attention buffers must not be null");
    }
    if (request.query_tokens == 0 || request.key_tokens == 0 || request.heads == 0 ||
        request.head_dim == 0) {
        return cpp_llm_engine::core::Status::error("attention dimensions must be non-zero");
    }
    if (!std::isfinite(request.scale) || request.scale < 0.0F) {
        return cpp_llm_engine::core::Status::error(
            "attention scale must be finite and non-negative");
    }
    if (request.heads > std::numeric_limits<std::size_t>::max() / request.head_dim) {
        return cpp_llm_engine::core::Status::error("attention head width overflows host size");
    }
    const std::size_t width = request.heads * request.head_dim;
    if (request.query_tokens > std::numeric_limits<std::size_t>::max() / width ||
        request.key_tokens > std::numeric_limits<std::size_t>::max() / width) {
        return cpp_llm_engine::core::Status::error("attention buffer size overflows host size");
    }
    if (request.query_position_offset >
        std::numeric_limits<std::size_t>::max() - (request.query_tokens - 1)) {
        return cpp_llm_engine::core::Status::error("attention query position overflows host size");
    }

    const float scale = request.scale == 0.0f
                            ? 1.0f / std::sqrt(static_cast<float>(request.head_dim))
                            : request.scale;
    std::vector<float> scores(request.key_tokens, 0.0f);
    for (std::size_t query_token = 0; query_token < request.query_tokens; ++query_token) {
        for (std::size_t head = 0; head < request.heads; ++head) {
            const float* query_row = request.query + query_token * width + head * request.head_dim;
            for (std::size_t key_token = 0; key_token < request.key_tokens; ++key_token) {
                const std::size_t query_position = request.query_position_offset + query_token;
                if (request.causal && key_token > query_position) {
                    scores[key_token] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float* key_row = request.key + key_token * width + head * request.head_dim;
                float score = 0.0f;
                for (std::size_t dim = 0; dim < request.head_dim; ++dim) {
                    score += query_row[dim] * key_row[dim];
                }
                scores[key_token] = score * scale;
            }

            float max_score = -std::numeric_limits<float>::infinity();
            for (const float score : scores) max_score = std::max(max_score, score);
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::isinf(score) && score < 0.0f ? 0.0f : std::exp(score - max_score);
                denominator += score;
            }
            if (denominator == 0.0f) {
                return cpp_llm_engine::core::Status::error("attention has no visible key positions");
            }

            float* output_row = request.output + query_token * width + head * request.head_dim;
            for (std::size_t dim = 0; dim < request.head_dim; ++dim) {
                float value = 0.0f;
                for (std::size_t key_token = 0; key_token < request.key_tokens; ++key_token) {
                    const float* value_row = request.value + key_token * width + head * request.head_dim;
                    value += (scores[key_token] / denominator) * value_row[dim];
                }
                output_row[dim] = value;
            }
        }
    }
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace chapter04_attention
