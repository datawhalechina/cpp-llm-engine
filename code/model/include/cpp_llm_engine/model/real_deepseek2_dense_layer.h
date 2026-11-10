// Model：声明 DeepSeek2 稠密层与 MoE 层的 CPU 执行入口。
// 职责：复用 RMSNorm、partial RoPE、压缩 MLA 和双残差，替换 FFN 分支。
// 契约：处理组合 attn_kv_b；失败时不覆盖结果或请求 Context。
#pragma once

#include "chapter05_quantization/quantized_linear_policy.h"
#include "model_metadata.h"
#include "real_layer_weights.h"
#include "cpp_llm_engine/model/real_moe_feed_forward.h"

#include <string>
#include <vector>

namespace cpp_llm_engine::model {

struct RealDeepSeek2LayerResult {
    std::vector<float> attention_norm;
    std::vector<float> query;
    std::vector<float> compressed_kv_with_rope;
    std::vector<float> normalized_latent_kv;
    std::vector<float> attention_value;
    std::vector<float> attention_output;
    std::vector<float> attention_residual;
    std::vector<float> ffn_norm;
    std::vector<float> ffn_output;
    std::vector<float> output;
};

struct RealDeepSeek2LayerContext {
    std::size_t position = 0;
    std::vector<float> latent_kv;
    std::vector<float> key_rope;

    void reset();
};

using RealDeepSeek2DenseLayerResult = RealDeepSeek2LayerResult;
using RealDeepSeek2DenseLayerContext = RealDeepSeek2LayerContext;

bool run_deepseek2_dense_token_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2DenseLayerWeights& weights,
    const std::vector<float>& input,
    RealDeepSeek2LayerContext* context,
    RealDeepSeek2LayerResult* result,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {});

bool run_deepseek2_dense_first_token_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2DenseLayerWeights& weights,
    const std::vector<float>& input,
    RealDeepSeek2LayerResult* result,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {});

bool run_deepseek2_moe_token_layer(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2MoeLayerWeights& weights,
    const std::vector<float>& input,
    RealDeepSeek2LayerContext* context,
    RealDeepSeek2LayerResult* result,
    RealMoeRoute* route,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {},
    RealMoeTrace* trace = nullptr);

bool run_deepseek2_moe_token_layer_batch(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2MoeLayerWeights& weights,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<RealDeepSeek2LayerContext*>& contexts,
    std::vector<RealDeepSeek2LayerResult>* results,
    std::vector<RealMoeRoute>* routes,
    RealMoeDispatchStats* stats,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {},
    RealMoeBatchMode mode = RealMoeBatchMode::Grouped);

}  // namespace cpp_llm_engine::model
