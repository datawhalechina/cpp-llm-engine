// Model：声明 Qwen2MoE 单层 CPU 执行入口。
// 职责：连接 RMSNorm、NeoX RoPE、因果 Attention、MoE FFN 与残差。
// 契约：按输入顺序生成位置，并返回旋转后的 K/V 负载。
// 验证：qwen15_moe_ffn_compare 使用 llama.cpp 逐层中间张量对照。
#pragma once

#include "cpp_llm_engine/model/real_moe_feed_forward.h"

#include <string>
#include <vector>

namespace cpp_llm_engine::model {

struct RealQwen2MoeLayerResult {
    std::vector<float> attention_residual;
    std::vector<float> moe_output;
    std::vector<float> output;
    RealMoeRoute route;
};

struct RealQwen2MoePrefillResult {
    std::vector<RealQwen2MoeLayerResult> tokens;
    std::vector<float> keys;
    std::vector<float> values;
};

struct RealQwen2MoeLayerContext {
    std::size_t position = 0;
    std::vector<float> keys;
    std::vector<float> values;

    void reset();
};

bool run_qwen2_moe_first_token_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<float>& input,
    RealQwen2MoeLayerResult* result,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {});

bool run_qwen2_moe_prefill_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    RealQwen2MoePrefillResult* result,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {});

bool run_qwen2_moe_layer_with_context(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    RealQwen2MoeLayerContext* context,
    RealQwen2MoePrefillResult* result,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {});

// 每个输入属于独立请求；Attention 保持各自 K/V，MoE 在请求之间统一分发。
bool run_qwen2_moe_layer_batch_with_context(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<RealQwen2MoeLayerContext*>& contexts,
    std::vector<RealQwen2MoeLayerResult>* results,
    RealMoeDispatchStats* dispatch_stats = nullptr,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {},
    RealMoeBatchMode batch_mode = RealMoeBatchMode::Grouped);

}  // namespace cpp_llm_engine::model
