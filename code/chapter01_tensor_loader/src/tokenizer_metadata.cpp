// Chapter 01：GGUF tokenizer metadata 适配实现
// 职责：读取 token、score、token type、BOS/EOS 和 chat template metadata。
// 契约：数组长度和 special token 类型不一致时返回失败。
// 验证：chapter01_tensor_loader_test.cpp 与真实 TinyLlama inspect。
#include "tokenizer_metadata.h"

#include <limits>

namespace chapter01 {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

bool read_token_id(const MetadataMap& values,
                   const std::string& key,
                   TokenId* output,
                   std::string* error) {
    const auto it = values.find(key);
    if (it == values.end()) {
        set_error(error, "missing tokenizer metadata key: " + key);
        return false;
    }
    std::int64_t number = 0;
    if (const auto* signed_value = std::get_if<std::int64_t>(&it->second)) {
        number = *signed_value;
    } else if (const auto* unsigned_value = std::get_if<std::uint64_t>(&it->second)) {
        if (*unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<TokenId>::max())) {
            set_error(error, "tokenizer token id is too large: " + key);
            return false;
        }
        number = static_cast<std::int64_t>(*unsigned_value);
    } else {
        set_error(error, "tokenizer token id is not an integer: " + key);
        return false;
    }
    if (number < -1 || number > std::numeric_limits<TokenId>::max()) {
        set_error(error, "tokenizer token id is out of range: " + key);
        return false;
    }
    *output = static_cast<TokenId>(number);
    return true;
}

bool read_optional_token_id(const MetadataMap& values,
                            const std::string& key,
                            TokenId* output,
                            std::string* error) {
    if (values.find(key) == values.end()) {
        *output = -1;
        return true;
    }
    return read_token_id(values, key, output, error);
}

}  // namespace

bool TokenizerMetadata::valid() const {
    return !tokens.empty() && (scores.empty() || scores.size() == tokens.size()) &&
           (token_types.empty() || token_types.size() == tokens.size());
}

bool TokenizerMetadataAdapter::from_gguf(const MetadataMap& values,
                                         TokenizerMetadata* metadata,
                                         std::string* error) {
    if (metadata == nullptr) {
        set_error(error, "tokenizer metadata output must not be null");
        return false;
    }
    TokenizerMetadata parsed;
    const auto tokens = values.find("tokenizer.ggml.tokens");
    if (tokens == values.end()) {
        set_error(error, "missing tokenizer metadata key: tokenizer.ggml.tokens");
        return false;
    }
    const auto* token_list = std::get_if<std::vector<std::string>>(&tokens->second);
    if (token_list == nullptr) {
        set_error(error, "tokenizer.ggml.tokens is not a string array");
        return false;
    }
    parsed.tokens = *token_list;

    if (const auto it = values.find("tokenizer.ggml.scores"); it != values.end()) {
        const auto* scores = std::get_if<std::vector<float>>(&it->second);
        if (scores == nullptr) {
            set_error(error, "tokenizer.ggml.scores is not a float array");
            return false;
        }
        parsed.scores = *scores;
    }
    if (const auto it = values.find("tokenizer.ggml.token_type"); it != values.end()) {
        if (const auto* token_types = std::get_if<std::vector<std::uint32_t>>(&it->second)) {
            parsed.token_types = *token_types;
        } else if (const auto* signed_token_types = std::get_if<std::vector<std::int32_t>>(&it->second)) {
            parsed.token_types.reserve(signed_token_types->size());
            for (const auto token_type : *signed_token_types) {
                if (token_type < 0) {
                    set_error(error, "tokenizer.ggml.token_type contains a negative value");
                    return false;
                }
                parsed.token_types.push_back(static_cast<std::uint32_t>(token_type));
            }
        } else {
            set_error(error, "tokenizer.ggml.token_type is not an integer array");
            return false;
        }
    }
    if (const auto it = values.find("tokenizer.chat_template"); it != values.end()) {
        const auto* chat_template = std::get_if<std::string>(&it->second);
        if (chat_template == nullptr) {
            set_error(error, "tokenizer.chat_template is not a string");
            return false;
        }
        parsed.chat_template = *chat_template;
    }
    if (!read_optional_token_id(values, "tokenizer.ggml.bos_token_id",
                                &parsed.bos_token_id, error) ||
        !read_token_id(values, "tokenizer.ggml.eos_token_id", &parsed.eos_token_id, error) ||
        !parsed.valid()) {
        if (error != nullptr && error->empty()) {
            *error = "tokenizer metadata arrays have inconsistent lengths";
        }
        return false;
    }
    *metadata = std::move(parsed);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter01
