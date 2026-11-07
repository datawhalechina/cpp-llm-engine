// Chapter 03：Tensor View 到矩阵乘请求的适配实现
// 流程：校验二维 View 的设备、类型和布局，再构造连续行主序 MatmulRequest。
// 提交：先写入临时结果，计算成功后再复制到输出；输入不会被隐式连续化。
// 验证：bash code/chapter03_matmul/scripts/run.sh
#include "chapter03_matmul/tensor_matmul_adapter.h"

#include "cpp_llm_engine/core/backend.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace chapter03_matmul {

namespace {

using cpp_llm_engine::core::BackendKind;
using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::DeviceKind;
using cpp_llm_engine::core::MutableTensorViewRef;
using cpp_llm_engine::core::Status;
using cpp_llm_engine::core::TensorViewRef;

Status validate_cpu_matrix_view(const TensorViewRef& view) {
    const auto storage_status = view.validate(sizeof(float));
    if (!storage_status.is_ok()) return storage_status;
    if (view.contract.shape.dimensions.size() != 2 ||
        view.contract.strides.size() != 2) {
        return Status::error("matmul tensor view rank must be two");
    }
    if (view.contract.dtype != DType::Float32) {
        return Status::error("matmul tensor view dtype must be Float32");
    }
    if (view.contract.device != DeviceKind::CPU) {
        return Status::error("matmul tensor view device must be CPU");
    }
    if (!view.contract.is_contiguous_row_major()) {
        return Status::error("matmul requires contiguous row-major input");
    }
    return Status::ok();
}

const float* first_element(const TensorViewRef& view) {
    return static_cast<const float*>(view.storage_data) +
           view.contract.storage_offset_elements;
}

float* first_element(const MutableTensorViewRef& view) {
    return static_cast<float*>(view.storage_data) +
           view.contract.storage_offset_elements;
}

}  // namespace

cpp_llm_engine::core::Status try_compute_tensor_views_into(
    const TensorViewRef& lhs,
    const TensorViewRef& rhs,
    const cpp_llm_engine::core::MatmulBackend& backend,
    const MutableTensorViewRef& output) {
    if (backend.kind() != BackendKind::CPU) {
        return Status::error("tensor view adapter currently requires a CPU backend");
    }
    const auto lhs_status = validate_cpu_matrix_view(lhs);
    if (!lhs_status.is_ok()) return lhs_status;
    const auto rhs_status = validate_cpu_matrix_view(rhs);
    if (!rhs_status.is_ok()) return rhs_status;
    const auto output_status = validate_cpu_matrix_view(output.as_const());
    if (!output_status.is_ok()) return output_status;

    const auto rows = lhs.contract.shape.dimensions[0];
    const auto inner = lhs.contract.shape.dimensions[1];
    if (rhs.contract.shape.dimensions[0] != inner) {
        return Status::error("matmul shape mismatch");
    }
    const auto cols = rhs.contract.shape.dimensions[1];
    if (output.contract.shape.dimensions != std::vector<std::size_t>{rows, cols}) {
        return Status::error("matmul output shape mismatch");
    }

    cpp_llm_engine::core::Shape output_shape{{rows, cols}};
    std::size_t output_elements = 0;
    const auto shape_status = output_shape.checked_numel(&output_elements);
    if (!shape_status.is_ok()) return shape_status;
    std::vector<float> result(output_elements, 0.0f);
    const cpp_llm_engine::core::MatmulRequest request{
        first_element(lhs), first_element(rhs), result.data(), rows, inner, cols};
    const auto compute_status = backend.compute(request);
    if (!compute_status.is_ok()) return compute_status;
    std::copy(result.begin(), result.end(), first_element(output));
    return Status::ok();
}

cpp_llm_engine::core::Status try_compute_tensor_views(
    const cpp_llm_engine::core::TensorViewRef& lhs,
    const cpp_llm_engine::core::TensorViewRef& rhs,
    const cpp_llm_engine::core::MatmulBackend& backend,
    Matrix* output) {
    if (output == nullptr) {
        return Status::error("matmul output must not be null");
    }
    if (backend.kind() != BackendKind::CPU) {
        return Status::error("tensor view adapter currently requires a CPU backend");
    }
    const auto lhs_status = validate_cpu_matrix_view(lhs);
    if (!lhs_status.is_ok()) return lhs_status;
    const auto rhs_status = validate_cpu_matrix_view(rhs);
    if (!rhs_status.is_ok()) return rhs_status;
    if (rhs.contract.shape.dimensions[0] != lhs.contract.shape.dimensions[1]) {
        return Status::error("matmul shape mismatch");
    }
    const auto rows = lhs.contract.shape.dimensions[0];
    const auto cols = rhs.contract.shape.dimensions[1];
    Matrix result(rows, cols, 0.0f);
    const cpp_llm_engine::core::TensorContract result_contract{
        cpp_llm_engine::core::Shape{{rows, cols}},
        {cols, 1},
        DType::Float32,
        DeviceKind::CPU,
        0,
        result.data.size() * sizeof(float),
    };
    const MutableTensorViewRef result_view{result.data.data(), result_contract};
    const auto compute_status =
        try_compute_tensor_views_into(lhs, rhs, backend, result_view);
    if (!compute_status.is_ok()) return compute_status;
    *output = std::move(result);
    return Status::ok();
}

}  // namespace chapter03_matmul
