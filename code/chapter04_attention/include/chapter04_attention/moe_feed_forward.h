// Chapter 04：教学型 MoE FeedForward 接口
// 职责：暴露 Router、稳定 Top-K、专家 SwiGLU 和共享专家合并结果。
// 契约：输入输出保持 [tokens, model_dim]，失败不覆盖调用方结果。
// 验证：moe_feed_forward_test.cpp。
#pragma once

#include "chapter04_attention/tensor.h"
#include "cpp_llm_engine/core/status.h"

#include <cstddef>
#include <vector>

namespace chapter04_attention {

struct MoeExpertWeights {
    Tensor gate;
    Tensor up;
    Tensor down;

    bool valid(std::size_t model_dim, std::size_t hidden_dim) const;
};

struct MoeRoute {
    std::vector<std::size_t> expert_indices;
    std::vector<float> weights;
};

struct MoeForwardResult {
    Tensor output;
    std::vector<MoeRoute> routes;

    bool ready() const;
};

class MoeFeedForward final {
public:
    MoeFeedForward(std::size_t model_dim,
                   std::size_t expert_hidden_dim,
                   std::size_t top_k,
                   Tensor router_weight,
                   std::vector<MoeExpertWeights> experts,
                   std::vector<float> shared_expert_router,
                   MoeExpertWeights shared_expert);

    MoeForwardResult forward(const Tensor& input) const;
    cpp_llm_engine::core::Status try_forward(
        const Tensor& input, MoeForwardResult* result) const;
    bool ready() const;

    std::size_t expert_count() const;
    std::size_t top_k() const;

private:
    std::size_t model_dim_ = 0;
    std::size_t expert_hidden_dim_ = 0;
    std::size_t top_k_ = 0;
    Tensor router_weight_;
    std::vector<MoeExpertWeights> experts_;
    std::vector<float> shared_expert_router_;
    MoeExpertWeights shared_expert_;
};

}  // namespace chapter04_attention
