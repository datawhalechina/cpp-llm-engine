// Chapter 01：模型 metadata 对象接口
// 职责：保存与文件格式无关的模型架构和 tokenizer 关键字段。
// 契约：对象是 ModelConfig、ModelSchema 和 tokenizer adapter 的输入。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include "tokenizer.h"

#include <cstddef>
#include <string>

namespace chapter01 {

struct ModelMetadata {
    std::string architecture;
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t max_position = 0;
    std::size_t layer_count = 0;
    std::size_t attention_head_count = 0;
    std::size_t attention_head_count_kv = 0;
    std::size_t attention_key_length = 0;
    std::size_t attention_value_length = 0;
    std::size_t attention_key_length_mla = 0;
    std::size_t attention_value_length_mla = 0;
    std::size_t attention_kv_lora_rank = 0;
    std::size_t feed_forward_length = 0;
    std::size_t leading_dense_layer_count = 0;
    std::size_t expert_count = 0;
    std::size_t expert_used_count = 0;
    std::size_t expert_shared_count = 0;
    std::size_t expert_feed_forward_length = 0;
    std::size_t expert_shared_feed_forward_length = 0;
    bool expert_weights_normalize = true;
    float expert_weights_scale = 1.0f;
    std::size_t rope_dimension_count = 0;
    float rope_freq_base = 10000.0f;
    std::string rope_scaling_type;
    float rope_scaling_factor = 1.0f;
    std::size_t rope_scaling_original_context = 0;
    float rope_yarn_log_multiplier = 0.0f;
    float rms_norm_epsilon = 1.0e-5f;
    TokenId bos_token_id = -1;
    TokenId eos_token_id = -1;

    bool valid() const;
};

}  // namespace chapter01
