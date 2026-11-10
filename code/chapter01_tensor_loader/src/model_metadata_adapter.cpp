// Chapter 01：GGUF metadata 适配实现
// 职责：把 GGUF key/value metadata 转成稳定的 ModelMetadata 对象。
// 契约：缺少架构、维度或 special token 时返回明确错误。
// 验证：chapter01_tensor_loader_test.cpp 与真实 TinyLlama inspect。
#include "model_metadata_adapter.h"

#include <cmath>
#include <limits>

namespace chapter01 {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

const MetadataValue* find_value(const MetadataMap& values, const std::string& key) {
    const auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
}

bool read_string(const MetadataMap& values,
                 const std::string& key,
                 std::string* output,
                 std::string* error) {
    const MetadataValue* value = find_value(values, key);
    if (value == nullptr) {
        set_error(error, "missing metadata key: " + key);
        return false;
    }
    const auto* text = std::get_if<std::string>(value);
    if (text == nullptr || text->empty()) {
        set_error(error, "metadata key is not a non-empty string: " + key);
        return false;
    }
    *output = *text;
    return true;
}

bool read_optional_string(const MetadataMap& values,
                          const std::string& key,
                          std::string* output,
                          std::string* error) {
    const MetadataValue* value = find_value(values, key);
    if (value == nullptr) return true;
    const auto* text = std::get_if<std::string>(value);
    if (text == nullptr || text->empty()) {
        set_error(error, "metadata key is not a non-empty string: " + key);
        return false;
    }
    *output = *text;
    return true;
}

bool read_size(const MetadataMap& values,
               const std::string& key,
               std::size_t* output,
               std::string* error) {
    const MetadataValue* value = find_value(values, key);
    if (value == nullptr) {
        set_error(error, "missing metadata key: " + key);
        return false;
    }

    std::uint64_t number = 0;
    if (const auto* unsigned_value = std::get_if<std::uint64_t>(value)) {
        number = *unsigned_value;
    } else if (const auto* signed_value = std::get_if<std::int64_t>(value)) {
        if (*signed_value <= 0) {
            set_error(error, "metadata key must be positive: " + key);
            return false;
        }
        number = static_cast<std::uint64_t>(*signed_value);
    } else {
        set_error(error, "metadata key is not an integer: " + key);
        return false;
    }

    if (number > std::numeric_limits<std::size_t>::max()) {
        set_error(error, "metadata key is too large: " + key);
        return false;
    }
    *output = static_cast<std::size_t>(number);
    return true;
}

bool read_token_id(const MetadataMap& values,
                   const std::string& key,
                   TokenId* output,
                   std::string* error) {
    const MetadataValue* value = find_value(values, key);
    if (value == nullptr) {
        set_error(error, "missing metadata key: " + key);
        return false;
    }

    std::int64_t number = 0;
    if (const auto* signed_value = std::get_if<std::int64_t>(value)) {
        number = *signed_value;
    } else if (const auto* unsigned_value = std::get_if<std::uint64_t>(value)) {
        if (*unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<TokenId>::max())) {
            set_error(error, "token id is too large: " + key);
            return false;
        }
        number = static_cast<std::int64_t>(*unsigned_value);
    } else {
        set_error(error, "token id is not an integer: " + key);
        return false;
    }

    if (number < -1 || number > std::numeric_limits<TokenId>::max()) {
        set_error(error, "token id is out of range: " + key);
        return false;
    }
    *output = static_cast<TokenId>(number);
    return true;
}

bool read_optional_token_id(const MetadataMap& values,
                            const std::string& key,
                            TokenId* output,
                            std::string* error) {
    if (find_value(values, key) == nullptr) {
        *output = -1;
        return true;
    }
    return read_token_id(values, key, output, error);
}

bool read_optional_positive_float(const MetadataMap& values,
                                  const std::string& key,
                                  float* output,
                                  std::string* error) {
    const MetadataValue* value = find_value(values, key);
    if (value == nullptr) return true;
    const auto* number = std::get_if<float>(value);
    if (number == nullptr || !std::isfinite(*number) || *number <= 0.0f) {
        set_error(error, "metadata key is not a positive float: " + key);
        return false;
    }
    *output = *number;
    return true;
}

bool read_optional_positive_size(const MetadataMap& values,
                                 const std::string& key,
                                 std::size_t* output,
                                 std::string* error) {
    const MetadataValue* value = find_value(values, key);
    if (value == nullptr) return true;
    std::uint64_t number = 0;
    if (const auto* unsigned_value = std::get_if<std::uint64_t>(value)) {
        number = *unsigned_value;
    } else if (const auto* signed_value = std::get_if<std::int64_t>(value)) {
        if (*signed_value <= 0) {
            set_error(error, "metadata key must be positive: " + key);
            return false;
        }
        number = static_cast<std::uint64_t>(*signed_value);
    } else {
        set_error(error, "metadata key is not an integer: " + key);
        return false;
    }
    if (number == 0 || number > std::numeric_limits<std::size_t>::max()) {
        set_error(error, "metadata key is out of range: " + key);
        return false;
    }
    *output = static_cast<std::size_t>(number);
    return true;
}

bool read_vocab_size(const MetadataMap& values,
                     std::size_t* output,
                     std::string* error) {
    if (read_size(values, "tokenizer.ggml.vocab_size", output, error) ||
        read_size(values, "general.vocab_size", output, error)) {
        return true;
    }
    const auto tokens = values.find("tokenizer.ggml.tokens");
    if (tokens != values.end()) {
        if (const auto* token_list = std::get_if<std::vector<std::string>>(&tokens->second)) {
            if (token_list->size() > 1) {
                *output = token_list->size();
                return true;
            }
        }
    }
    set_error(error, "missing GGUF vocabulary size and tokenizer token array");
    return false;
}

}  // namespace

bool ModelMetadataAdapter::from_gguf(const MetadataMap& values,
                                     ModelMetadata* metadata,
                                     std::string* error) {
    if (metadata == nullptr) {
        set_error(error, "metadata output must not be null");
        return false;
    }

    ModelMetadata parsed;
    if (!read_string(values, "general.architecture", &parsed.architecture, error)) {
        return false;
    }

    const std::string prefix = parsed.architecture + ".";
    if (!read_vocab_size(values, &parsed.vocab_size, error)) {
        return false;
    }
    if (!read_size(values, prefix + "embedding_length", &parsed.hidden_size, error) ||
        !read_size(values, prefix + "context_length", &parsed.max_position, error) ||
        !read_size(values, prefix + "block_count", &parsed.layer_count, error) ||
        !read_optional_positive_size(values, prefix + "attention.head_count",
                                     &parsed.attention_head_count, error) ||
        !read_optional_positive_size(values, prefix + "attention.head_count_kv",
                                     &parsed.attention_head_count_kv, error) ||
        !read_optional_positive_size(values, prefix + "attention.key_length",
                                     &parsed.attention_key_length, error) ||
        !read_optional_positive_size(values, prefix + "attention.value_length",
                                     &parsed.attention_value_length, error) ||
        !read_optional_positive_size(values, prefix + "attention.key_length_mla",
                                     &parsed.attention_key_length_mla, error) ||
        !read_optional_positive_size(values, prefix + "attention.value_length_mla",
                                     &parsed.attention_value_length_mla, error) ||
        !read_optional_positive_size(values, prefix + "attention.kv_lora_rank",
                                     &parsed.attention_kv_lora_rank, error) ||
        !read_optional_positive_size(values, prefix + "feed_forward_length",
                                     &parsed.feed_forward_length, error) ||
        !read_optional_positive_size(values, prefix + "leading_dense_block_count",
                                     &parsed.leading_dense_layer_count, error) ||
        !read_optional_positive_size(values, prefix + "expert_count",
                                     &parsed.expert_count, error) ||
        !read_optional_positive_size(values, prefix + "expert_used_count",
                                     &parsed.expert_used_count, error) ||
        !read_optional_positive_size(values, prefix + "expert_shared_count",
                                     &parsed.expert_shared_count, error) ||
        !read_optional_positive_size(values, prefix + "expert_feed_forward_length",
                                     &parsed.expert_feed_forward_length, error) ||
        !read_optional_positive_size(values, prefix + "expert_shared_feed_forward_length",
                                     &parsed.expert_shared_feed_forward_length, error) ||
        !read_optional_positive_size(values, prefix + "rope.dimension_count",
                                     &parsed.rope_dimension_count, error) ||
        !read_optional_positive_float(values, prefix + "rope.freq_base",
                                      &parsed.rope_freq_base, error) ||
        !read_optional_string(values, prefix + "rope.scaling.type",
                              &parsed.rope_scaling_type, error) ||
        !read_optional_positive_float(values, prefix + "rope.scaling.factor",
                                      &parsed.rope_scaling_factor, error) ||
        !read_optional_positive_size(
            values, prefix + "rope.scaling.original_context_length",
            &parsed.rope_scaling_original_context, error) ||
        !read_optional_positive_float(
            values, prefix + "rope.scaling.yarn_log_multiplier",
            &parsed.rope_yarn_log_multiplier, error) ||
        !read_optional_positive_float(values, prefix + "attention.layer_norm_rms_epsilon",
                                      &parsed.rms_norm_epsilon, error) ||
        !read_optional_token_id(values, "tokenizer.ggml.bos_token_id",
                                &parsed.bos_token_id, error) ||
        !read_token_id(values, "tokenizer.ggml.eos_token_id", &parsed.eos_token_id, error)) {
        return false;
    }

    // Older DeepSeek2 GGUFs store expanded MLA head lengths in key/value_length
    // and omit the explicit *_mla keys. Normalize both generations here.
    if (parsed.architecture == "deepseek2" &&
        parsed.attention_key_length_mla == 0 &&
        parsed.attention_value_length_mla == 0 &&
        parsed.attention_kv_lora_rank > 0 &&
        parsed.rope_dimension_count > 0) {
        parsed.attention_key_length_mla = parsed.attention_key_length;
        parsed.attention_value_length_mla = parsed.attention_value_length;
        parsed.attention_key_length =
            parsed.attention_kv_lora_rank + parsed.rope_dimension_count;
        parsed.attention_value_length = parsed.attention_kv_lora_rank;
    }

    if (!parsed.valid()) {
        set_error(error, "parsed GGUF metadata is incomplete");
        return false;
    }
    // Qwen2MoE and DeepSeek2 keep probabilities from the full-router Softmax.
    // Other teaching paths retain selected-expert normalization by default.
    if (parsed.architecture == "qwen2moe" ||
        parsed.architecture == "deepseek2") {
        parsed.expert_weights_normalize = false;
    }
    *metadata = parsed;
    if (error != nullptr) {
        error->clear();
    }
    return true;
}

}  // namespace chapter01
