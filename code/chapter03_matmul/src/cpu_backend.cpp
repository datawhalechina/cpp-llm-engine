// Chapter 03：CPU 矩阵乘后端实现
// 行为：提供 CPU 后端身份，并把 MatmulRequest 交给参考内核执行。
// 错误：本文件不重复校验和计算，直接传回参考内核的 Status。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#include "chapter03_matmul/cpu_backend.h"
#include "chapter03_matmul/cpu_kernel.h"

namespace chapter03_matmul {

cpp_llm_engine::core::BackendKind CpuReferenceMatmulBackend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view CpuReferenceMatmulBackend::name() const {
    return "cpu-reference-matmul";
}

cpp_llm_engine::core::Status CpuReferenceMatmulBackend::compute(
    const cpp_llm_engine::core::MatmulRequest& request) const {
    return cpu_reference_matmul_kernel(request);
}

const CpuReferenceMatmulBackend& cpu_reference_backend() {
    static const CpuReferenceMatmulBackend backend;
    return backend;
}

}  // namespace chapter03_matmul
