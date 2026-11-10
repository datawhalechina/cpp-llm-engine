// Chapter 04：多头潜在注意力（MLA）CPU 数值参考。
// 低秩 K/V 与共享 RoPE Key 是持久状态，完整各头 K/V 只作为临时工作区。
// 验证：mla_attention_test.cpp 对照显式展开的标准多头 Attention。
#include "chapter04_attention/mla_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace chapter04_attention {
namespace {

using cpp_llm_engine::core::Status;

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t* result) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    *result = left * right;
    return true;
}

Status validate(const MlaAttentionRequest& request) {
    if (request.query_nope == nullptr || request.query_rope == nullptr ||
        request.latent_kv == nullptr || request.key_rope == nullptr ||
        request.key_up_weight == nullptr || request.value_up_weight == nullptr ||
        request.output == nullptr) {
        return Status::error("MLA buffers must not be null");
    }
    if (request.query_tokens == 0 || request.key_tokens == 0 ||
        request.heads == 0 || request.qk_nope_dim == 0 ||
        request.rope_dim == 0 || request.latent_dim == 0 ||
        request.value_dim == 0) {
        return Status::error("MLA dimensions must be non-zero");
    }
    if (!std::isfinite(request.scale) || request.scale < 0.0F) {
        return Status::error("MLA scale must be finite and non-negative");
    }
    if (request.causal &&
        (request.query_position_offset >= request.key_tokens ||
         request.query_tokens - 1 >
             request.key_tokens - 1 - request.query_position_offset)) {
        return Status::error("MLA query positions exceed the K/V history");
    }

    std::size_t head_key_width = 0;
    std::size_t head_value_width = 0;
    std::size_t key_elements = 0;
    std::size_t value_elements = 0;
    std::size_t output_elements = 0;
    if (!checked_multiply(request.heads, request.qk_nope_dim,
                          &head_key_width) ||
        !checked_multiply(request.heads, request.value_dim,
                          &head_value_width) ||
        !checked_multiply(request.key_tokens, head_key_width,
                          &key_elements) ||
        !checked_multiply(request.key_tokens, head_value_width,
                          &value_elements) ||
        !checked_multiply(request.query_tokens, head_value_width,
                          &output_elements)) {
        return Status::error("MLA workspace size overflows host size");
    }
    return Status::ok();
}

float dot(const float* left, const float* right, std::size_t count) {
    float result = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
        result += left[index] * right[index];
    }
    return result;
}

}  // namespace

cpp_llm_engine::core::Status cpu_mla_attention_reference(
    const MlaAttentionRequest& request) {
    const auto validation = validate(request);
    if (!validation.is_ok()) return validation;

    const std::size_t expanded_key_width =
        request.heads * request.qk_nope_dim;
    const std::size_t expanded_value_width =
        request.heads * request.value_dim;
    std::vector<float> expanded_keys(
        request.key_tokens * expanded_key_width, 0.0F);
    std::vector<float> expanded_values(
        request.key_tokens * expanded_value_width, 0.0F);

    for (std::size_t token = 0; token < request.key_tokens; ++token) {
        const float* latent = request.latent_kv + token * request.latent_dim;
        for (std::size_t head = 0; head < request.heads; ++head) {
            for (std::size_t dim = 0; dim < request.qk_nope_dim; ++dim) {
                const float* weight = request.key_up_weight +
                    (head * request.qk_nope_dim + dim) * request.latent_dim;
                expanded_keys[token * expanded_key_width +
                              head * request.qk_nope_dim + dim] =
                    dot(weight, latent, request.latent_dim);
            }
            for (std::size_t dim = 0; dim < request.value_dim; ++dim) {
                const float* weight = request.value_up_weight +
                    (head * request.value_dim + dim) * request.latent_dim;
                expanded_values[token * expanded_value_width +
                                head * request.value_dim + dim] =
                    dot(weight, latent, request.latent_dim);
            }
        }
    }

    const float scale = request.scale == 0.0F
        ? 1.0F / std::sqrt(static_cast<float>(
              request.qk_nope_dim + request.rope_dim))
        : request.scale;
    std::vector<float> scores(request.key_tokens, 0.0F);
    std::vector<float> result(
        request.query_tokens * expanded_value_width, 0.0F);
    for (std::size_t query_token = 0;
         query_token < request.query_tokens; ++query_token) {
        const std::size_t query_position =
            request.query_position_offset + query_token;
        const std::size_t visible_keys = request.causal
            ? query_position + 1
            : request.key_tokens;
        for (std::size_t head = 0; head < request.heads; ++head) {
            const float* query_nope = request.query_nope +
                (query_token * request.heads + head) * request.qk_nope_dim;
            const float* query_rope = request.query_rope +
                (query_token * request.heads + head) * request.rope_dim;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t key_token = 0;
                 key_token < visible_keys; ++key_token) {
                const float* key_nope = expanded_keys.data() +
                    key_token * expanded_key_width +
                    head * request.qk_nope_dim;
                const float* key_rope = request.key_rope +
                    key_token * request.rope_dim;
                scores[key_token] =
                    (dot(query_nope, key_nope, request.qk_nope_dim) +
                     dot(query_rope, key_rope, request.rope_dim)) * scale;
                maximum = std::max(maximum, scores[key_token]);
            }

            float denominator = 0.0F;
            for (std::size_t key_token = 0;
                 key_token < visible_keys; ++key_token) {
                scores[key_token] = std::exp(scores[key_token] - maximum);
                denominator += scores[key_token];
            }
            if (!std::isfinite(denominator) || denominator <= 0.0F) {
                return Status::error("MLA Softmax denominator is invalid");
            }

            float* output = result.data() +
                (query_token * request.heads + head) * request.value_dim;
            for (std::size_t key_token = 0;
                 key_token < visible_keys; ++key_token) {
                const float probability = scores[key_token] / denominator;
                const float* value = expanded_values.data() +
                    key_token * expanded_value_width +
                    head * request.value_dim;
                for (std::size_t dim = 0; dim < request.value_dim; ++dim) {
                    output[dim] += probability * value[dim];
                }
            }
        }
    }
    std::copy(result.begin(), result.end(), request.output);
    return Status::ok();
}

}  // namespace chapter04_attention
