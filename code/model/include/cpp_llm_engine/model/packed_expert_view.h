// Model：声明 GGUF 三维 packed expert 的只读二维权重 View。
// 职责：把指定 expert 的连续字节窗口映射为统一 Linear 权重描述。
// 契约：Tensor shape 为 [cols, rows, experts]，storage 生命周期覆盖 View。
// 验证：packed_expert_view_test 覆盖格式、偏移、边界和失败原子性。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"
#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace cpp_llm_engine::model {

struct PackedExpertView {
    const std::uint8_t* data = nullptr;
    std::size_t bytes = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    cpp_llm_engine::core::WeightFormat format =
        cpp_llm_engine::core::WeightFormat::Unknown;

    static bool create(const chapter01::Tensor& packed,
                       std::size_t expert_index,
                       PackedExpertView* output,
                       std::string* error = nullptr);
};

}  // namespace cpp_llm_engine::model
