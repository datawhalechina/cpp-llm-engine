// Model：声明真实量化 MoE FeedForward 的权重适配器。
// 职责：连接 Chapter04 路由语义、packed expert View 与 Chapter05 Linear。
// 契约：输入是单词元归一化隐藏向量，输出是待加入残差的 MoE 增量。
// 验证：real_moe_feed_forward_test 覆盖 Top-K、专家执行和共享专家合并。
#pragma once

#include "chapter05_quantization/quantized_linear_policy.h"
#include "model_metadata.h"
#include "real_layer_weights.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cpp_llm_engine::model {

struct RealMoeRoute {
    std::vector<std::size_t> expert_indices;
    std::vector<float> weights;
    float shared_expert_weight = 0.0F;
};

struct RealMoeTrace {
    std::vector<float> routed_output;
    std::vector<float> weighted_shared_output;
};

struct RealMoeDispatchStats {
    std::size_t assignment_count = 0;
    std::size_t active_expert_count = 0;
    std::size_t maximum_tokens_per_expert = 0;
};

enum class RealMoeBatchMode : std::uint8_t {
    RowWise = 0,
    Grouped,
};

bool run_real_moe_feed_forward(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<float>& normalized_hidden,
    std::vector<float>* output,
    RealMoeRoute* route,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {},
    RealMoeTrace* trace = nullptr);

bool run_real_deepseek2_moe_feed_forward(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2MoeLayerWeights& weights,
    const std::vector<float>& normalized_hidden,
    std::vector<float>* output,
    RealMoeRoute* route,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {},
    RealMoeTrace* trace = nullptr);

bool run_real_moe_feed_forward_batch(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& normalized_hidden,
    std::vector<std::vector<float>>* output,
    std::vector<RealMoeRoute>* routes,
    RealMoeDispatchStats* stats = nullptr,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {},
    RealMoeBatchMode mode = RealMoeBatchMode::Grouped);

bool run_real_deepseek2_moe_feed_forward_batch(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2MoeLayerWeights& weights,
    const std::vector<std::vector<float>>& normalized_hidden,
    std::vector<std::vector<float>>* output,
    std::vector<RealMoeRoute>* routes,
    RealMoeDispatchStats* stats = nullptr,
    std::string* error = nullptr,
    const chapter05_quantization::QuantizedLinearPolicy& policy = {},
    RealMoeBatchMode mode = RealMoeBatchMode::Grouped);

}  // namespace cpp_llm_engine::model
