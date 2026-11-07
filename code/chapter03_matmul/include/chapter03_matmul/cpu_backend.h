// Chapter 03：CPU 矩阵乘后端接口
// 输入输出：compute 接收行主序 MatmulRequest，并将结果写入其 output。
// 行为：报告 CPU 设备类型，把实际计算交给可读的参考内核。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#pragma once

#include "cpp_llm_engine/core/matmul_contract.h"

namespace chapter03_matmul {

class CpuReferenceMatmulBackend final : public cpp_llm_engine::core::MatmulBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::MatmulRequest& request) const override;
};

const CpuReferenceMatmulBackend& cpu_reference_backend();

}  // namespace chapter03_matmul
