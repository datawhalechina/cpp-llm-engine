// CUDA Backend: verify batched Router, stable Top-K and selected normalization.
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed: " #condition << " at line " << __LINE__ \
                      << '\n';                                                  \
            return 1;                                                           \
        }                                                                       \
    } while (false)

namespace {

struct Route final {
    std::vector<std::size_t> indices;
    std::vector<float> weights;
};

Route cpu_route(
    const float* hidden,
    const std::vector<float>& router,
    std::size_t hidden_size,
    std::size_t experts,
    std::size_t top_k,
    bool normalize,
    float scale) {
    std::vector<float> logits(experts, 0.0F);
    for (std::size_t expert = 0; expert < experts; ++expert) {
        for (std::size_t column = 0; column < hidden_size; ++column) {
            logits[expert] += hidden[column] *
                router[expert * hidden_size + column];
        }
    }
    std::vector<std::size_t> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&logits](auto lhs, auto rhs) {
        if (logits[lhs] == logits[rhs]) return lhs < rhs;
        return logits[lhs] > logits[rhs];
    });
    order.resize(top_k);
    const float maximum = *std::max_element(logits.begin(), logits.end());
    float denominator = 0.0F;
    for (const float logit : logits) denominator += std::exp(logit - maximum);
    std::vector<float> weights(top_k);
    float selected_sum = 0.0F;
    for (std::size_t slot = 0; slot < top_k; ++slot) {
        weights[slot] = std::exp(logits[order[slot]] - maximum) / denominator;
        selected_sum += weights[slot];
    }
    for (float& weight : weights) {
        weight *= normalize ? scale / selected_sum : scale;
    }
    return {std::move(order), std::move(weights)};
}

}  // namespace

int main() {
    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;

    constexpr std::size_t tokens = 3;
    constexpr std::size_t hidden_size = 5;
    constexpr std::size_t experts = 4;
    constexpr std::size_t top_k = 2;
    constexpr float scale = 1.7F;
    const std::vector<float> hidden{
        0.2F, -0.5F, 0.7F, 0.1F, -0.3F,
        -0.4F, 0.6F, 0.2F, -0.8F, 0.5F,
        1.0F, 0.0F, -0.25F, 0.5F, 0.75F};
    const std::vector<float> router{
        0.1F, 0.2F, 0.3F, 0.4F, 0.5F,
        -0.3F, 0.7F, 0.2F, -0.1F, 0.6F,
        0.5F, -0.2F, 0.4F, 0.8F, -0.6F,
        0.1F, 0.2F, 0.3F, 0.4F, 0.5F};

    CudaExecutionContext context;
    CudaTensor hidden_device;
    CudaTensor router_device;
    CudaTensor indices_device;
    CudaTensor weights_device;
    CudaTensor grouped_rows_device;
    CudaTensor grouped_weights_device;
    CudaTensor expert_counts_device;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {tokens, hidden_size}, DType::Float32, &hidden_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {experts, hidden_size}, DType::Float32, &router_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {tokens, top_k}, DType::Float32, &indices_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {tokens, top_k}, DType::Float32, &weights_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {experts, tokens * top_k}, DType::Float32,
        &grouped_rows_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {experts, tokens * top_k}, DType::Float32,
        &grouped_weights_device).is_ok());
    CHECK(CudaTensor::create_contiguous(
        {experts}, DType::Float32, &expert_counts_device).is_ok());
    CHECK(context.upload(
        &hidden_device, hidden.data(), hidden.size() * sizeof(float)).is_ok());
    CHECK(context.upload(
        &router_device, router.data(), router.size() * sizeof(float)).is_ok());
    CHECK(context.moe_route(
        hidden_device, router_device, top_k, true, scale,
        &indices_device, &weights_device).is_ok());
    CHECK(context.moe_group_routes(
        indices_device, weights_device, experts, &grouped_rows_device,
        &grouped_weights_device, &expert_counts_device).is_ok());

    std::vector<float> indices(tokens * top_k);
    std::vector<float> weights(tokens * top_k);
    std::vector<float> grouped_rows(experts * tokens * top_k);
    std::vector<float> grouped_weights(experts * tokens * top_k);
    std::vector<float> expert_counts(experts);
    CHECK(context.download(
        indices_device, indices.data(), indices.size() * sizeof(float)).is_ok());
    CHECK(context.download(
        weights_device, weights.data(), weights.size() * sizeof(float)).is_ok());
    CHECK(context.download(grouped_rows_device, grouped_rows.data(),
        grouped_rows.size() * sizeof(float)).is_ok());
    CHECK(context.download(grouped_weights_device, grouped_weights.data(),
        grouped_weights.size() * sizeof(float)).is_ok());
    CHECK(context.download(expert_counts_device, expert_counts.data(),
        expert_counts.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    std::vector<std::vector<std::pair<std::size_t, float>>> expected_groups(
        experts);
    for (std::size_t token = 0; token < tokens; ++token) {
        const Route expected = cpu_route(
            hidden.data() + token * hidden_size, router, hidden_size,
            experts, top_k, true, scale);
        for (std::size_t slot = 0; slot < top_k; ++slot) {
            const std::size_t offset = token * top_k + slot;
            CHECK(static_cast<std::size_t>(indices[offset]) ==
                  expected.indices[slot]);
            CHECK(std::fabs(weights[offset] - expected.weights[slot]) <= 1.0e-6F);
            expected_groups[expected.indices[slot]].push_back(
                {token, expected.weights[slot]});
        }
    }
    const std::size_t capacity = tokens * top_k;
    for (std::size_t expert = 0; expert < experts; ++expert) {
        CHECK(static_cast<std::size_t>(expert_counts[expert]) ==
              expected_groups[expert].size());
        for (std::size_t item = 0; item < expected_groups[expert].size(); ++item) {
            const std::size_t offset = expert * capacity + item;
            CHECK(static_cast<std::size_t>(grouped_rows[offset]) ==
                  expected_groups[expert][item].first);
            CHECK(std::fabs(grouped_weights[offset] -
                  expected_groups[expert][item].second) <= 1.0e-6F);
        }
    }
    CHECK(!context.moe_route(
        hidden_device, router_device, experts + 1, true, scale,
        &indices_device, &weights_device).is_ok());
    CHECK(!context.moe_group_routes(
        indices_device, weights_device, experts + 1, &grouped_rows_device,
        &grouped_weights_device, &expert_counts_device).is_ok());
    std::cout << "cuda_moe_router_test: PASS\n";
    return 0;
}
