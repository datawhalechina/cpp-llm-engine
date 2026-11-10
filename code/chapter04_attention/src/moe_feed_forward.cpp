// Chapter 04：教学型 MoE FeedForward 实现
// 职责：按 token 完成路由、专家计算和共享专家合并。
// 契约：Top-K 分数相同时优先较小专家编号，路由权重只在选中专家间归一化。
// 验证：moe_feed_forward_test.cpp。
#include "chapter04_attention/moe_feed_forward.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace chapter04_attention {
namespace {

using cpp_llm_engine::core::Status;

float sigmoid(float value) {
    if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

float silu(float value) {
    return value * sigmoid(value);
}

Status run_expert(const Tensor& input,
                  std::size_t token,
                  const MoeExpertWeights& expert,
                  std::vector<float>* output) {
    const std::size_t model_dim = input.cols;
    const std::size_t hidden_dim = expert.gate.cols;
    std::vector<float> hidden(hidden_dim, 0.0F);
    for (std::size_t column = 0; column < hidden_dim; ++column) {
        float gate = 0.0F;
        float up = 0.0F;
        for (std::size_t inner = 0; inner < model_dim; ++inner) {
            const float value = input.at(token, inner);
            gate += value * expert.gate.at(inner, column);
            up += value * expert.up.at(inner, column);
        }
        hidden[column] = silu(gate) * up;
    }

    output->assign(model_dim, 0.0F);
    for (std::size_t column = 0; column < model_dim; ++column) {
        for (std::size_t inner = 0; inner < hidden_dim; ++inner) {
            (*output)[column] += hidden[inner] * expert.down.at(inner, column);
        }
        if (!std::isfinite((*output)[column])) {
            return Status::error("MoE expert produced a non-finite value");
        }
    }
    return Status::ok();
}

}  // namespace

bool MoeExpertWeights::valid(std::size_t model_dim,
                             std::size_t hidden_dim) const {
    return gate.ready() && up.ready() && down.ready() &&
           gate.rows == model_dim && gate.cols == hidden_dim &&
           up.rows == model_dim && up.cols == hidden_dim &&
           down.rows == hidden_dim && down.cols == model_dim;
}

bool MoeForwardResult::ready() const {
    return output.ready() && routes.size() == output.rows;
}

MoeFeedForward::MoeFeedForward(
    std::size_t model_dim,
    std::size_t expert_hidden_dim,
    std::size_t top_k,
    Tensor router_weight,
    std::vector<MoeExpertWeights> experts,
    std::vector<float> shared_expert_router,
    MoeExpertWeights shared_expert)
    : model_dim_(model_dim),
      expert_hidden_dim_(expert_hidden_dim),
      top_k_(top_k),
      router_weight_(std::move(router_weight)),
      experts_(std::move(experts)),
      shared_expert_router_(std::move(shared_expert_router)),
      shared_expert_(std::move(shared_expert)) {}

MoeForwardResult MoeFeedForward::forward(const Tensor& input) const {
    MoeForwardResult result;
    if (!try_forward(input, &result).is_ok()) return {};
    return result;
}

Status MoeFeedForward::try_forward(const Tensor& input,
                                   MoeForwardResult* result) const {
    if (result == nullptr) return Status::error("MoE result must not be null");
    if (!ready()) return Status::error("MoE feed forward is not ready");
    if (!input.ready() || input.cols != model_dim_) {
        return Status::error("MoE input shape mismatch");
    }

    MoeForwardResult built;
    built.output = Tensor(input.rows, model_dim_);
    built.routes.reserve(input.rows);
    for (std::size_t token = 0; token < input.rows; ++token) {
        std::vector<float> logits(experts_.size(), 0.0F);
        for (std::size_t expert = 0; expert < experts_.size(); ++expert) {
            for (std::size_t inner = 0; inner < model_dim_; ++inner) {
                logits[expert] += input.at(token, inner) *
                                  router_weight_.at(inner, expert);
            }
            if (!std::isfinite(logits[expert])) {
                return Status::error("MoE router produced a non-finite value");
            }
        }

        std::vector<std::size_t> order(experts_.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&logits](std::size_t left, std::size_t right) {
                             if (logits[left] == logits[right]) return left < right;
                             return logits[left] > logits[right];
                         });
        order.resize(top_k_);

        const float maximum = logits[order.front()];
        std::vector<float> route_weights(top_k_, 0.0F);
        float denominator = 0.0F;
        for (std::size_t selected = 0; selected < top_k_; ++selected) {
            route_weights[selected] =
                std::exp(logits[order[selected]] - maximum);
            denominator += route_weights[selected];
        }
        if (!std::isfinite(denominator) || denominator <= 0.0F) {
            return Status::error("MoE route normalization failed");
        }
        for (float& weight : route_weights) weight /= denominator;

        for (std::size_t selected = 0; selected < top_k_; ++selected) {
            std::vector<float> expert_output;
            auto status = run_expert(
                input, token, experts_[order[selected]], &expert_output);
            if (!status.is_ok()) return status;
            for (std::size_t column = 0; column < model_dim_; ++column) {
                built.output.at(token, column) +=
                    route_weights[selected] * expert_output[column];
            }
        }

        float shared_logit = 0.0F;
        for (std::size_t inner = 0; inner < model_dim_; ++inner) {
            shared_logit += input.at(token, inner) * shared_expert_router_[inner];
        }
        if (!std::isfinite(shared_logit)) {
            return Status::error("MoE shared expert router produced a non-finite value");
        }
        std::vector<float> shared_output;
        auto status = run_expert(input, token, shared_expert_, &shared_output);
        if (!status.is_ok()) return status;
        const float shared_weight = sigmoid(shared_logit);
        for (std::size_t column = 0; column < model_dim_; ++column) {
            built.output.at(token, column) += shared_weight * shared_output[column];
            if (!std::isfinite(built.output.at(token, column))) {
                return Status::error("MoE output is not finite");
            }
        }
        built.routes.push_back({std::move(order), std::move(route_weights)});
    }

    *result = std::move(built);
    return Status::ok();
}

bool MoeFeedForward::ready() const {
    if (model_dim_ == 0 || expert_hidden_dim_ == 0 || experts_.empty() ||
        top_k_ == 0 || top_k_ > experts_.size() ||
        !router_weight_.ready() || router_weight_.rows != model_dim_ ||
        router_weight_.cols != experts_.size() ||
        shared_expert_router_.size() != model_dim_ ||
        !shared_expert_.valid(model_dim_, expert_hidden_dim_)) {
        return false;
    }
    return std::all_of(
        experts_.begin(), experts_.end(),
        [this](const MoeExpertWeights& expert) {
            return expert.valid(model_dim_, expert_hidden_dim_);
        });
}

std::size_t MoeFeedForward::expert_count() const {
    return experts_.size();
}

std::size_t MoeFeedForward::top_k() const {
    return top_k_;
}

}  // namespace chapter04_attention
