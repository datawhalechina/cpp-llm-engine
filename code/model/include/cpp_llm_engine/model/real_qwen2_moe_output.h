// Model：声明 Qwen2MoE 最终归一化与 LM Head 执行入口。
// 职责：把最后一层隐藏状态转换为归一化状态和词表 logits。
// 契约：输出权重形状为 [hidden,vocab]，输出长度等于词表大小。
// 验证：qwen15_moe_ffn_compare 使用 llama.cpp result_norm/result_output 对照。
#pragma once

#include "chapter05_quantization/quantized_linear_policy.h"
#include "model_metadata.h"
#include "tensor.h"

#include <string>
#include <vector>

namespace cpp_llm_engine::model {

bool finish_quantized_model_output(
    const chapter01::ModelMetadata& metadata,
    const chapter01::Tensor& output_norm,
    const chapter01::Tensor& output_weight,
    const std::vector<float>& hidden,
    std::vector<float>* normalized_hidden,
    std::vector<float>* logits,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {});

bool finish_qwen2_moe_output(
    const chapter01::ModelMetadata& metadata,
    const chapter01::Tensor& output_norm,
    const chapter01::Tensor& output_weight,
    const std::vector<float>& hidden,
    std::vector<float>* normalized_hidden,
    std::vector<float>* logits,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {});

}  // namespace cpp_llm_engine::model
