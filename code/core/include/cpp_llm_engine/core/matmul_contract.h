// Core：矩阵乘后端请求
// 输入输出：调用方提供行主序 lhs[M,K]、rhs[K,N] 和 output[M,N] 缓冲区。
// 限制：三个缓冲区不得重叠且容量必须满足形状；后端只能校验指针、维度和溢出。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#pragma once

#include "backend.h"
#include "status.h"

#include <cstddef>

namespace cpp_llm_engine::core {

struct MatmulRequest final {
    const float* lhs = nullptr;
    const float* rhs = nullptr;
    float* output = nullptr;
    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t cols = 0;
};

class MatmulBackend : public Backend {
public:
    ~MatmulBackend() override = default;
    virtual Status compute(const MatmulRequest& request) const = 0;
};

}  // namespace cpp_llm_engine::core
