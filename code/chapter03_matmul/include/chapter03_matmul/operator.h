// Chapter 03：教学算子接口
// 输入输出：compute 接收左右两个 Matrix，返回新 Matrix。
// 设计：基类固定调用形式，具体算子选择计算后端并实现计算过程。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#pragma once

#include "chapter03_matmul/matrix.h"
#include "cpp_llm_engine/core/operator.h"

namespace chapter03_matmul {

class Operator : public cpp_llm_engine::core::Operator {
public:
    virtual ~Operator() = default;
    virtual Matrix compute(const Matrix& lhs, const Matrix& rhs) const = 0;
};

}  // namespace chapter03_matmul
