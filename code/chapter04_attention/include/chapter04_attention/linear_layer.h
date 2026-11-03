// Chapter 04：Decoder 内部 Linear 接口
// 职责：提供 Attention projection 和 FFN 使用的教学 float Linear。
// 契约：输入列数为 in_features，输出列数为 out_features；失败保留可写 View。
// 验证：chapter04_attention_test.cpp。
#pragma once

#include "chapter04_attention/tensor.h"
#include "cpp_llm_engine/core/status.h"
#include "cpp_llm_engine/core/tensor_view_ref.h"

#include <cstddef>
#include <vector>

namespace chapter04_attention {

class LinearLayer final {
public:
    LinearLayer(std::size_t in_features = 0, std::size_t out_features = 0);

    Tensor forward(const Tensor& input) const;
    cpp_llm_engine::core::Status try_forward(
        const cpp_llm_engine::core::TensorViewRef& input,
        const cpp_llm_engine::core::MutableTensorViewRef& output) const;
    bool ready() const;

    std::size_t in_features() const;
    std::size_t out_features() const;

private:
    std::size_t in_features_ = 0;
    std::size_t out_features_ = 0;
    Tensor weight_;
    std::vector<float> bias_;
};

}  // namespace chapter04_attention
