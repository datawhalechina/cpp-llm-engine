// Model：实现真实量化 MoE FeedForward 的按需专家执行。
// 职责：先完成 Router/Top-K，再只切片并执行命中的专家和共享专家。
// 契约：所有输出在完整成功后一次提交，失败不修改调用方结果。
// 验证：real_moe_feed_forward_test。
#include "cpp_llm_engine/model/real_moe_feed_forward.h"

#include "cpp_llm_engine/model/packed_expert_view.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace cpp_llm_engine::model {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) error->assign(message);
    return false;
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t value = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            std::uint32_t normalized = mantissa;
            std::uint32_t adjusted = 113;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --adjusted;
            }
            value = sign | (adjusted << 23) |
                ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float converted = 0.0F;
    std::memcpy(&converted, &value, sizeof(converted));
    return converted;
}

bool floating_values(const chapter01::Tensor& tensor,
                     std::vector<float>* output,
                     std::string* error) {
    if (output == nullptr || tensor.raw_data() == nullptr) {
        return fail(error, "floating tensor output and payload must be present");
    }
    std::vector<float> values(tensor.numel(), 0.0F);
    if (tensor.dtype() == chapter01::DType::Float32) {
        std::memcpy(values.data(), tensor.raw_data(), values.size() * sizeof(float));
    } else if (tensor.dtype() == chapter01::DType::Float16) {
        const auto* source =
            reinterpret_cast<const std::uint16_t*>(tensor.raw_data());
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = half_to_float(source[i]);
        }
    } else {
        return fail(error, "Router tensor must use Float32 or Float16");
    }
    *output = std::move(values);
    return true;
}

bool quantized_linear(const void* data,
                      std::size_t bytes,
                      std::size_t rows,
                      std::size_t cols,
                      cpp_llm_engine::core::WeightFormat format,
                      const std::vector<float>& input,
                      const chapter05_quantization::QuantizedLinearPolicy& policy,
                      std::vector<float>* output,
                      std::string* error) {
    std::vector<float> values(rows, 0.0F);
    const cpp_llm_engine::core::LinearRequest request{
        data, bytes, input.data(), input.size(), values.data(), values.size(),
        rows, cols, format};
    const auto status =
        chapter05_quantization::compute_quantized_linear(request, policy);
    if (!status.is_ok()) return fail(error, status.message());
    *output = std::move(values);
    return true;
}

bool quantized_linear_batch(
    const void* data,
    std::size_t bytes,
    std::size_t rows,
    std::size_t cols,
    cpp_llm_engine::core::WeightFormat format,
    const std::vector<std::vector<float>>& inputs,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<std::vector<float>>* outputs,
    std::string* error) {
    if (outputs == nullptr || inputs.empty() ||
        std::any_of(inputs.begin(), inputs.end(),
                    [cols](const std::vector<float>& input) {
                        return input.size() != cols;
                    })) {
        return fail(error, "MoE Linear batch input shape is inconsistent");
    }
    std::vector<float> packed_inputs;
    packed_inputs.reserve(inputs.size() * cols);
    for (const auto& input : inputs) {
        packed_inputs.insert(packed_inputs.end(), input.begin(), input.end());
    }
    std::vector<float> packed_outputs(inputs.size() * rows, 0.0F);
    const cpp_llm_engine::core::LinearBatchRequest request{
        data, bytes, packed_inputs.data(), packed_inputs.size(),
        packed_outputs.data(), packed_outputs.size(), inputs.size(), rows,
        cols, format};
    const auto status =
        chapter05_quantization::compute_quantized_linear_batch(request, policy);
    if (!status.is_ok()) return fail(error, status.message());
    std::vector<std::vector<float>> built(
        inputs.size(), std::vector<float>(rows, 0.0F));
    for (std::size_t batch = 0; batch < inputs.size(); ++batch) {
        std::copy_n(packed_outputs.begin() + batch * rows, rows,
                    built[batch].begin());
    }
    *outputs = std::move(built);
    return true;
}

bool tensor_view(const chapter01::Tensor& weight,
                 PackedExpertView* view,
                 std::string* error) {
    const auto& shape = weight.shape();
    if (view == nullptr || shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
        return fail(error, "shared expert weight must be a positive 2D tensor");
    }
    PackedExpertView built;
    built.data = reinterpret_cast<const std::uint8_t*>(weight.raw_data());
    built.bytes = weight.bytes();
    built.cols = static_cast<std::size_t>(shape[0]);
    built.rows = static_cast<std::size_t>(shape[1]);
    if (weight.dtype() == chapter01::DType::Q2_K) {
        built.format = cpp_llm_engine::core::WeightFormat::Q2K;
    } else if (weight.dtype() == chapter01::DType::Q3_K) {
        built.format = cpp_llm_engine::core::WeightFormat::Q3K;
    } else if (weight.dtype() == chapter01::DType::Q4_K) {
        built.format = cpp_llm_engine::core::WeightFormat::Q4K;
    } else if (weight.dtype() == chapter01::DType::Q5_0) {
        built.format = cpp_llm_engine::core::WeightFormat::Q5_0;
    } else if (weight.dtype() == chapter01::DType::Q5_1) {
        built.format = cpp_llm_engine::core::WeightFormat::Q5_1;
    } else if (weight.dtype() == chapter01::DType::Q5_K) {
        built.format = cpp_llm_engine::core::WeightFormat::Q5K;
    } else if (weight.dtype() == chapter01::DType::Q6_K) {
        built.format = cpp_llm_engine::core::WeightFormat::Q6K;
    } else if (weight.dtype() == chapter01::DType::Q8_0) {
        built.format = cpp_llm_engine::core::WeightFormat::Q8_0;
    } else if (weight.dtype() == chapter01::DType::IQ4_NL) {
        built.format = cpp_llm_engine::core::WeightFormat::IQ4_NL;
    } else {
        return fail(error, "shared expert weight format is not supported: " +
            weight.name() + " dtype=" +
            std::to_string(static_cast<int>(weight.dtype())));
    }
    *view = built;
    return true;
}

bool tensor_linear(
    const chapter01::Tensor& weight,
    const std::vector<float>& input,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<float>* output,
    std::string* error) {
    PackedExpertView view;
    if (!tensor_view(weight, &view, error)) return false;
    return quantized_linear(view.data, view.bytes, view.rows, view.cols,
                            view.format, input, policy, output, error);
}

bool tensor_linear_batch(
    const chapter01::Tensor& weight,
    const std::vector<std::vector<float>>& inputs,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<std::vector<float>>* outputs,
    std::string* error) {
    PackedExpertView view;
    if (!tensor_view(weight, &view, error)) return false;
    return quantized_linear_batch(
        view.data, view.bytes, view.rows, view.cols, view.format, inputs,
        policy, outputs, error);
}

float sigmoid(float value) {
    if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

struct MoeWeightsView {
    const chapter01::Tensor& router;
    const chapter01::Tensor& expert_gate;
    const chapter01::Tensor& expert_up;
    const chapter01::Tensor& expert_down;
    const chapter01::Tensor& shared_expert_gate;
    const chapter01::Tensor& shared_expert_up;
    const chapter01::Tensor& shared_expert_down;
    const chapter01::Tensor* shared_expert_router = nullptr;
};

MoeWeightsView weights_view(const chapter01::RealMoeLayerWeights& weights) {
    return {weights.router, weights.expert_gate, weights.expert_up,
            weights.expert_down, weights.shared_expert_gate,
            weights.shared_expert_up, weights.shared_expert_down,
            &weights.shared_expert_router};
}

MoeWeightsView weights_view(
    const chapter01::RealDeepSeek2MoeLayerWeights& weights) {
    return {weights.router, weights.expert_gate, weights.expert_up,
            weights.expert_down, weights.shared_expert_gate,
            weights.shared_expert_up, weights.shared_expert_down, nullptr};
}

bool run_packed_expert(
    const MoeWeightsView& weights,
    std::size_t expert,
    const std::vector<float>& input,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<float>* output,
    std::string* error) {
    PackedExpertView gate;
    PackedExpertView up;
    PackedExpertView down;
    if (!PackedExpertView::create(weights.expert_gate, expert, &gate, error) ||
        !PackedExpertView::create(weights.expert_up, expert, &up, error) ||
        !PackedExpertView::create(weights.expert_down, expert, &down, error)) {
        return false;
    }
    std::vector<float> gate_values;
    std::vector<float> up_values;
    if (!quantized_linear(gate.data, gate.bytes, gate.rows, gate.cols,
                          gate.format, input, policy, &gate_values, error) ||
        !quantized_linear(up.data, up.bytes, up.rows, up.cols,
                          up.format, input, policy, &up_values, error)) {
        return false;
    }
    for (std::size_t i = 0; i < gate_values.size(); ++i) {
        gate_values[i] = gate_values[i] * sigmoid(gate_values[i]) * up_values[i];
    }
    return quantized_linear(down.data, down.bytes, down.rows, down.cols,
                            down.format, gate_values, policy, output, error);
}

struct PreparedExpert {
    PackedExpertView gate;
    PackedExpertView up;
    PackedExpertView down;
};

bool prepare_expert(const MoeWeightsView& weights,
                    std::size_t expert,
                    PreparedExpert* output,
                    std::string* error) {
    PreparedExpert prepared;
    if (!PackedExpertView::create(
            weights.expert_gate, expert, &prepared.gate, error) ||
        !PackedExpertView::create(
            weights.expert_up, expert, &prepared.up, error) ||
        !PackedExpertView::create(
            weights.expert_down, expert, &prepared.down, error)) {
        return false;
    }
    *output = prepared;
    return true;
}

bool run_prepared_expert(
    const PreparedExpert& expert,
    const std::vector<float>& input,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<float>* output,
    std::string* error) {
    std::vector<float> gate_values;
    std::vector<float> up_values;
    if (!quantized_linear(
            expert.gate.data, expert.gate.bytes, expert.gate.rows,
            expert.gate.cols, expert.gate.format, input, policy,
            &gate_values, error) ||
        !quantized_linear(
            expert.up.data, expert.up.bytes, expert.up.rows,
            expert.up.cols, expert.up.format, input, policy,
            &up_values, error)) {
        return false;
    }
    for (std::size_t i = 0; i < gate_values.size(); ++i) {
        gate_values[i] = gate_values[i] * sigmoid(gate_values[i]) *
            up_values[i];
    }
    return quantized_linear(
        expert.down.data, expert.down.bytes, expert.down.rows,
        expert.down.cols, expert.down.format, gate_values, policy,
        output, error);
}

bool run_prepared_expert_batch(
    const PreparedExpert& expert,
    const std::vector<std::vector<float>>& inputs,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<std::vector<float>>* outputs,
    std::string* error) {
    std::vector<std::vector<float>> gate;
    std::vector<std::vector<float>> up;
    if (!quantized_linear_batch(
            expert.gate.data, expert.gate.bytes, expert.gate.rows,
            expert.gate.cols, expert.gate.format, inputs, policy, &gate,
            error) ||
        !quantized_linear_batch(
            expert.up.data, expert.up.bytes, expert.up.rows, expert.up.cols,
            expert.up.format, inputs, policy, &up, error)) {
        return false;
    }
    for (std::size_t batch = 0; batch < gate.size(); ++batch) {
        for (std::size_t column = 0; column < gate[batch].size(); ++column) {
            gate[batch][column] = gate[batch][column] *
                sigmoid(gate[batch][column]) * up[batch][column];
        }
    }
    return quantized_linear_batch(
        expert.down.data, expert.down.bytes, expert.down.rows,
        expert.down.cols, expert.down.format, gate, policy, outputs, error);
}

bool run_shared_expert(
    const MoeWeightsView& weights,
    const std::vector<float>& input,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<float>* output,
    std::string* error) {
    std::vector<float> gate;
    std::vector<float> up;
    if (!tensor_linear(weights.shared_expert_gate, input, policy, &gate, error) ||
        !tensor_linear(weights.shared_expert_up, input, policy, &up, error)) {
        return false;
    }
    for (std::size_t i = 0; i < gate.size(); ++i) {
        gate[i] = gate[i] * sigmoid(gate[i]) * up[i];
    }
    return tensor_linear(weights.shared_expert_down, gate, policy, output, error);
}

bool run_shared_expert_batch(
    const MoeWeightsView& weights,
    const std::vector<std::vector<float>>& inputs,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    std::vector<std::vector<float>>* outputs,
    std::string* error) {
    std::vector<std::vector<float>> gate;
    std::vector<std::vector<float>> up;
    if (!tensor_linear_batch(
            weights.shared_expert_gate, inputs, policy, &gate, error) ||
        !tensor_linear_batch(
            weights.shared_expert_up, inputs, policy, &up, error)) {
        return false;
    }
    for (std::size_t batch = 0; batch < gate.size(); ++batch) {
        for (std::size_t column = 0; column < gate[batch].size(); ++column) {
            gate[batch][column] = gate[batch][column] *
                sigmoid(gate[batch][column]) * up[batch][column];
        }
    }
    return tensor_linear_batch(
        weights.shared_expert_down, gate, policy, outputs, error);
}

bool route_hidden(const chapter01::ModelMetadata& metadata,
                  const std::vector<float>& router,
                  const std::vector<float>* shared_router,
                  const std::vector<float>& hidden,
                  RealMoeRoute* route,
                  std::string* error) {
    std::vector<float> logits(metadata.expert_count, 0.0F);
    for (std::size_t expert = 0; expert < metadata.expert_count; ++expert) {
        for (std::size_t column = 0; column < metadata.hidden_size; ++column) {
            logits[expert] += router[expert * metadata.hidden_size + column] *
                hidden[column];
        }
        if (!std::isfinite(logits[expert])) {
            return fail(error, "MoE Router produced a non-finite score");
        }
    }
    std::vector<std::size_t> order(metadata.expert_count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&logits](std::size_t left, std::size_t right) {
                         if (logits[left] == logits[right]) return left < right;
                         return logits[left] > logits[right];
                     });
    order.resize(metadata.expert_used_count);

    const float maximum = *std::max_element(logits.begin(), logits.end());
    float denominator = 0.0F;
    for (const float logit : logits) denominator += std::exp(logit - maximum);
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
        return fail(error, "MoE selected expert weights cannot be normalized");
    }
    std::vector<float> route_weights(order.size(), 0.0F);
    float selected_sum = 0.0F;
    for (std::size_t i = 0; i < order.size(); ++i) {
        route_weights[i] = std::exp(logits[order[i]] - maximum) / denominator;
        selected_sum += route_weights[i];
    }
    if (metadata.expert_weights_normalize) {
        for (float& value : route_weights) value /= selected_sum;
    }
    for (float& value : route_weights) value *= metadata.expert_weights_scale;

    float shared_weight = 1.0F;
    if (shared_router != nullptr) {
        float shared_logit = 0.0F;
        for (std::size_t column = 0; column < metadata.hidden_size; ++column) {
            shared_logit += (*shared_router)[column] * hidden[column];
        }
        shared_weight = sigmoid(shared_logit);
    }
    *route = {std::move(order), std::move(route_weights),
              shared_weight};
    return true;
}

bool run_moe_feed_forward(
    const chapter01::ModelMetadata& metadata,
    const MoeWeightsView& weights,
    const std::vector<float>& normalized_hidden,
    std::vector<float>* output,
    RealMoeRoute* route,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeTrace* trace) {
    if (output == nullptr || route == nullptr) {
        return fail(error, "MoE output and route must not be null");
    }
    if (normalized_hidden.size() != metadata.hidden_size ||
        metadata.expert_count == 0 || metadata.expert_used_count == 0 ||
        metadata.expert_used_count > metadata.expert_count) {
        return fail(error, "MoE metadata and hidden input are inconsistent");
    }

    std::vector<float> router;
    std::vector<float> shared_router;
    if (!floating_values(weights.router, &router, error) ||
        router.size() != metadata.hidden_size * metadata.expert_count) {
        return fail(error, "MoE Router payload does not match metadata");
    }
    const std::vector<float>* shared_router_view = nullptr;
    if (weights.shared_expert_router != nullptr) {
        if (!floating_values(*weights.shared_expert_router, &shared_router,
                             error) ||
            shared_router.size() != metadata.hidden_size) {
            return fail(error,
                        "MoE shared Router payload does not match metadata");
        }
        shared_router_view = &shared_router;
    }

    RealMoeRoute built_route;
    if (!route_hidden(metadata, router, shared_router_view, normalized_hidden,
                      &built_route, error)) {
        return false;
    }

    std::vector<float> built(metadata.hidden_size, 0.0F);
    for (std::size_t i = 0; i < built_route.expert_indices.size(); ++i) {
        std::vector<float> expert_output;
        if (!run_packed_expert(weights, built_route.expert_indices[i],
                               normalized_hidden, policy, &expert_output,
                               error)) {
            return false;
        }
        for (std::size_t column = 0; column < built.size(); ++column) {
            built[column] += built_route.weights[i] * expert_output[column];
        }
    }

    std::vector<float> shared_output;
    if (!run_shared_expert(weights, normalized_hidden, policy, &shared_output,
                           error)) {
        return false;
    }
    std::vector<float> weighted_shared(shared_output.size(), 0.0F);
    for (std::size_t column = 0; column < built.size(); ++column) {
        weighted_shared[column] =
            built_route.shared_expert_weight * shared_output[column];
        built[column] += weighted_shared[column];
        if (!std::isfinite(built[column])) {
            return fail(error, "MoE FeedForward produced a non-finite value");
        }
    }

    *output = std::move(built);
    *route = std::move(built_route);
    if (trace != nullptr) {
        trace->routed_output.resize(output->size());
        for (std::size_t column = 0; column < output->size(); ++column) {
            trace->routed_output[column] =
                (*output)[column] - weighted_shared[column];
        }
        trace->weighted_shared_output = std::move(weighted_shared);
    }
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace

bool run_real_moe_feed_forward(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<float>& normalized_hidden,
    std::vector<float>* output,
    RealMoeRoute* route,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeTrace* trace) {
    if (metadata.architecture != "qwen2moe") {
        return fail(error, "Qwen2MoE metadata is required");
    }
    return run_moe_feed_forward(metadata, weights_view(weights),
                                normalized_hidden, output, route, error,
                                policy, trace);
}

bool run_real_deepseek2_moe_feed_forward(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2MoeLayerWeights& weights,
    const std::vector<float>& normalized_hidden,
    std::vector<float>* output,
    RealMoeRoute* route,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeTrace* trace) {
    if (metadata.architecture != "deepseek2") {
        return fail(error, "DeepSeek2 metadata is required");
    }
    return run_moe_feed_forward(metadata, weights_view(weights),
                                normalized_hidden, output, route, error,
                                policy, trace);
}

bool run_moe_feed_forward_batch_impl(
    const chapter01::ModelMetadata& metadata,
    const MoeWeightsView& weights,
    const std::vector<std::vector<float>>& normalized_hidden,
    std::vector<std::vector<float>>* output,
    std::vector<RealMoeRoute>* routes,
    RealMoeDispatchStats* stats,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeBatchMode mode) {
    if (output == nullptr || routes == nullptr || normalized_hidden.empty() ||
        metadata.expert_count == 0 || metadata.expert_used_count == 0 ||
        metadata.expert_used_count > metadata.expert_count) {
        return fail(error, "MoE batch inputs and metadata are inconsistent");
    }
    for (const auto& hidden : normalized_hidden) {
        if (hidden.size() != metadata.hidden_size) {
            return fail(error, "MoE batch hidden width is inconsistent");
        }
    }

    std::vector<float> router;
    std::vector<float> shared_router;
    if (!floating_values(weights.router, &router, error) ||
        router.size() != metadata.hidden_size * metadata.expert_count) {
        return fail(error, "MoE Router payload does not match metadata");
    }
    const std::vector<float>* shared_router_view = nullptr;
    if (weights.shared_expert_router != nullptr) {
        if (!floating_values(
                *weights.shared_expert_router, &shared_router, error) ||
            shared_router.size() != metadata.hidden_size) {
            return fail(error, "MoE shared Router payload does not match metadata");
        }
        shared_router_view = &shared_router;
    }

    std::vector<RealMoeRoute> built_routes(normalized_hidden.size());
    for (std::size_t token = 0; token < normalized_hidden.size(); ++token) {
        if (!route_hidden(metadata, router, shared_router_view,
                          normalized_hidden[token], &built_routes[token],
                          error)) {
            return false;
        }
    }

    struct Assignment {
        std::size_t token = 0;
        std::size_t route_index = 0;
    };
    std::vector<std::vector<Assignment>> assignments(metadata.expert_count);
    for (std::size_t token = 0; token < built_routes.size(); ++token) {
        for (std::size_t selected = 0;
             selected < built_routes[token].expert_indices.size(); ++selected) {
            assignments[built_routes[token].expert_indices[selected]].push_back(
                {token, selected});
        }
    }

    RealMoeDispatchStats built_stats;
    built_stats.assignment_count =
        normalized_hidden.size() * metadata.expert_used_count;
    std::vector<std::vector<float>> built(
        normalized_hidden.size(),
        std::vector<float>(metadata.hidden_size, 0.0F));
    for (std::size_t expert = 0; expert < assignments.size(); ++expert) {
        if (assignments[expert].empty()) continue;
        ++built_stats.active_expert_count;
        built_stats.maximum_tokens_per_expert = std::max(
            built_stats.maximum_tokens_per_expert, assignments[expert].size());
        PreparedExpert prepared;
        if (!prepare_expert(weights, expert, &prepared, error)) {
            return false;
        }
        std::vector<std::vector<float>> expert_outputs;
        if (mode == RealMoeBatchMode::Grouped) {
            std::vector<std::vector<float>> expert_inputs;
            expert_inputs.reserve(assignments[expert].size());
            for (const auto& assignment : assignments[expert]) {
                expert_inputs.push_back(normalized_hidden[assignment.token]);
            }
            if (!run_prepared_expert_batch(
                    prepared, expert_inputs, policy, &expert_outputs, error)) {
                return false;
            }
        } else if (mode == RealMoeBatchMode::RowWise) {
            expert_outputs.reserve(assignments[expert].size());
            for (const auto& assignment : assignments[expert]) {
                std::vector<float> expert_output;
                if (!run_prepared_expert(
                        prepared, normalized_hidden[assignment.token], policy,
                        &expert_output, error)) {
                    return false;
                }
                expert_outputs.push_back(std::move(expert_output));
            }
        } else {
            return fail(error, "MoE batch execution mode is invalid");
        }
        for (std::size_t item = 0; item < assignments[expert].size(); ++item) {
            const auto& assignment = assignments[expert][item];
            const float weight = built_routes[assignment.token]
                .weights[assignment.route_index];
            for (std::size_t column = 0; column < metadata.hidden_size;
                 ++column) {
                built[assignment.token][column] +=
                    weight * expert_outputs[item][column];
            }
        }
    }

    std::vector<std::vector<float>> shared_outputs;
    if (mode == RealMoeBatchMode::Grouped) {
        if (!run_shared_expert_batch(
                weights, normalized_hidden, policy,
                &shared_outputs, error)) {
            return false;
        }
    } else {
        shared_outputs.reserve(normalized_hidden.size());
        for (const auto& hidden : normalized_hidden) {
            std::vector<float> shared_output;
            if (!run_shared_expert(
                    weights, hidden, policy, &shared_output,
                    error)) {
                return false;
            }
            shared_outputs.push_back(std::move(shared_output));
        }
    }
    for (std::size_t token = 0; token < normalized_hidden.size(); ++token) {
        for (std::size_t column = 0; column < metadata.hidden_size; ++column) {
            built[token][column] +=
                built_routes[token].shared_expert_weight *
                shared_outputs[token][column];
            if (!std::isfinite(built[token][column])) {
                return fail(error,
                            "MoE batch FeedForward produced a non-finite value");
            }
        }
    }

    *output = std::move(built);
    *routes = std::move(built_routes);
    if (stats != nullptr) *stats = built_stats;
    if (error != nullptr) error->clear();
    return true;
}

bool run_real_moe_feed_forward_batch(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealMoeLayerWeights& weights,
    const std::vector<std::vector<float>>& normalized_hidden,
    std::vector<std::vector<float>>* output,
    std::vector<RealMoeRoute>* routes,
    RealMoeDispatchStats* stats,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeBatchMode mode) {
    if (metadata.architecture != "qwen2moe") {
        return fail(error, "Qwen2MoE metadata is required");
    }
    return run_moe_feed_forward_batch_impl(
        metadata, weights_view(weights), normalized_hidden, output, routes,
        stats, error, policy, mode);
}

bool run_real_deepseek2_moe_feed_forward_batch(
    const chapter01::ModelMetadata& metadata,
    const chapter01::RealDeepSeek2MoeLayerWeights& weights,
    const std::vector<std::vector<float>>& normalized_hidden,
    std::vector<std::vector<float>>* output,
    std::vector<RealMoeRoute>* routes,
    RealMoeDispatchStats* stats,
    std::string* error,
    const chapter05_quantization::QuantizedLinearPolicy& policy,
    RealMoeBatchMode mode) {
    if (metadata.architecture != "deepseek2") {
        return fail(error, "DeepSeek2 metadata is required");
    }
    return run_moe_feed_forward_batch_impl(
        metadata, weights_view(weights), normalized_hidden, output, routes,
        stats, error, policy, mode);
}

}  // namespace cpp_llm_engine::model
