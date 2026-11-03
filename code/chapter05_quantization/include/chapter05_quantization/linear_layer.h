// Chapter 05：声明浮点 Linear correctness baseline。
// 职责：提供与量化 Linear 共享形状语义的教学参考实现。
#pragma once

#include "chapter05_quantization/tensor.h"

#include <cstddef>
#include <vector>

namespace chapter05_quantization {

class LinearLayer final {
public:
    LinearLayer(std::size_t in_features = 0, std::size_t out_features = 0);

    Tensor forward(const Tensor& input) const;
    bool ready() const;

private:
    std::size_t in_features_ = 0;
    std::size_t out_features_ = 0;
    Tensor weight_;
    std::vector<float> bias_;
};

}  // namespace chapter05_quantization
