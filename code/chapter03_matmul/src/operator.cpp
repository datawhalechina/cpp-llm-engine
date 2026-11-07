// Chapter 03：教学算子接口编译单元
// 行为：只保留 Operator 的独立编译入口，不执行矩阵计算。
// 实现：具体算子在子类中实现 compute，并选择所需计算后端。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#include "chapter03_matmul/operator.h"

namespace chapter03_matmul {

}  // namespace chapter03_matmul
