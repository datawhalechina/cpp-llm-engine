// Chapter 04：FeedForward 接口
// 职责：声明 decoder FFN 的升维、激活和降维路径。
// 契约：输入列数为 model_dim，输出列数恢复为 model_dim。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "chapter04_attention/linear_layer.h"

#include <cstddef>

namespace chapter04_attention {

enum class ActivationKind {
    Relu,
};

class FeedForward final {
public:
    FeedForward(std::size_t model_dim = 0, std::size_t hidden_dim = 0);

    Tensor forward(const Tensor& input) const;
    cpp_llm_engine::core::Status try_forward(
        const cpp_llm_engine::core::TensorViewRef& input,
        const cpp_llm_engine::core::MutableTensorViewRef& output) const;
    bool ready() const;

    std::size_t model_dim() const;
    std::size_t hidden_dim() const;

private:
    std::size_t model_dim_ = 0;
    std::size_t hidden_dim_ = 0;
    LinearLayer fc1_;
    LinearLayer fc2_;
};

Tensor apply_activation(const Tensor& input, ActivationKind kind);
cpp_llm_engine::core::Status try_apply_activation(
    const cpp_llm_engine::core::TensorViewRef& input,
    ActivationKind kind,
    const cpp_llm_engine::core::MutableTensorViewRef& output);

}  // namespace chapter04_attention
