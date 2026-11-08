// CUDA 执行层：验证 Tensor、Workspace 和 ExecutionContext 的统一边界。
// 职责：覆盖 Tensor 复制、backend 调度、workspace 复用、移动和 shape 错误。
// 通过条件：全部 CHECK 成立，且上层测试不直接创建 DeviceBuffer 或 CUDA stream。
#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <utility>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed: " #condition << " at line " << __LINE__ \
                      << '\n';                                                  \
            return 1;                                                           \
        }                                                                       \
    } while (false)

int main() {
    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::core::DeviceKind;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;

    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count) == cudaSuccess);
    CHECK(device_count > 0);
    CHECK(!CudaExecutionContext::create(nullptr).is_ok());
    CHECK(!CudaTensor::create_contiguous({}, DType::Float32, nullptr).is_ok());

    CudaTensor invalid_tensor;
    CHECK(!CudaTensor::create_contiguous({}, DType::Float32, &invalid_tensor).is_ok());
    CHECK(!CudaTensor::create_contiguous({2, 0}, DType::Float32, &invalid_tensor).is_ok());
    CHECK(!CudaTensor::create_contiguous({2}, DType::Float16, &invalid_tensor).is_ok());
    CHECK(!CudaTensor::create_contiguous(
        {std::numeric_limits<std::size_t>::max(), 2},
        DType::Float32,
        &invalid_tensor).is_ok());

    CudaExecutionContext context;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(context.valid());
    CHECK(context.workspace_capacity_bytes() == 0);
    CHECK(context.workspace_allocation_count() == 0);
    CHECK(context.reserve_workspace(1024).is_ok());
    CHECK(context.workspace_capacity_bytes() == 1024);
    CHECK(context.workspace_allocation_count() == 1);
    CHECK(context.reserve_workspace(512).is_ok());
    CHECK(context.workspace_capacity_bytes() == 1024);
    CHECK(context.workspace_allocation_count() == 1);
    CHECK(context.reserve_workspace(2048).is_ok());
    CHECK(context.workspace_capacity_bytes() == 2048);
    CHECK(context.workspace_allocation_count() == 2);

    constexpr std::size_t elements = 257;
    std::array<float, elements> lhs_host{};
    std::array<float, elements> rhs_host{};
    std::array<float, elements> output_host{};
    for (std::size_t index = 0; index < elements; ++index) {
        lhs_host[index] = static_cast<float>(index);
        rhs_host[index] = 0.5F * static_cast<float>(index);
    }

    CudaTensor lhs;
    CudaTensor rhs;
    CudaTensor output;
    CHECK(CudaTensor::create_contiguous({elements}, DType::Float32, &lhs).is_ok());
    CHECK(CudaTensor::create_contiguous({elements}, DType::Float32, &rhs).is_ok());
    CHECK(CudaTensor::create_contiguous({elements}, DType::Float32, &output).is_ok());
    CHECK(lhs.contract().device == DeviceKind::CUDA);
    CHECK(lhs.contract().strides == std::vector<std::size_t>{1});
    CHECK(lhs.size_bytes() == sizeof(lhs_host));
    CHECK(!context.upload(&lhs, lhs_host.data(), sizeof(lhs_host) - 1).is_ok());
    CHECK(context.upload(&lhs, lhs_host.data(), sizeof(lhs_host)).is_ok());
    CHECK(context.upload(&rhs, rhs_host.data(), sizeof(rhs_host)).is_ok());
    CHECK(context.vector_add(lhs, rhs, &output).is_ok());
    CHECK(context.download(output, output_host.data(), sizeof(output_host)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < elements; ++index) {
        CHECK(output_host[index] == lhs_host[index] + rhs_host[index]);
    }

    CudaTensor copied;
    CHECK(CudaTensor::create_contiguous({elements}, DType::Float32, &copied).is_ok());
    CHECK(context.copy(output, &copied).is_ok());
    output_host.fill(0.0F);
    CHECK(context.download(copied, output_host.data(), sizeof(output_host)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < elements; ++index) {
        CHECK(output_host[index] == lhs_host[index] + rhs_host[index]);
    }
    CHECK(!context.copy(output, nullptr).is_ok());

    CudaTensor row_a;
    CudaTensor row_b;
    CudaTensor row_c;
    CudaTensor batch;
    for (CudaTensor* row : {&row_a, &row_b, &row_c}) {
        CHECK(CudaTensor::create_contiguous({4}, DType::Float32, row).is_ok());
    }
    CHECK(CudaTensor::create_contiguous({3, 4}, DType::Float32, &batch).is_ok());
    const std::array<float, 4> row_a_host{1, 2, 3, 4};
    const std::array<float, 4> row_b_host{5, 6, 7, 8};
    const std::array<float, 4> row_c_host{9, 10, 11, 12};
    CHECK(context.upload(&row_a, row_a_host.data(), sizeof(row_a_host)).is_ok());
    CHECK(context.upload(&row_b, row_b_host.data(), sizeof(row_b_host)).is_ok());
    CHECK(context.upload(&row_c, row_c_host.data(), sizeof(row_c_host)).is_ok());
    CHECK(context.pack_rows({&row_a, &row_b, &row_c}, &batch).is_ok());
    std::array<float, 12> batch_host{};
    CHECK(context.download(batch, batch_host.data(), sizeof(batch_host)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK((batch_host == std::array<float, 12>{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

    const std::array<float, 12> replacement_batch{
        12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    CHECK(context.upload(
        &batch, replacement_batch.data(), sizeof(replacement_batch)).is_ok());
    CHECK(context.unpack_rows(batch, {&row_a, &row_b, &row_c}).is_ok());
    std::array<float, 4> unpacked_a{};
    std::array<float, 4> unpacked_b{};
    std::array<float, 4> unpacked_c{};
    CHECK(context.download(row_a, unpacked_a.data(), sizeof(unpacked_a)).is_ok());
    CHECK(context.download(row_b, unpacked_b.data(), sizeof(unpacked_b)).is_ok());
    CHECK(context.download(row_c, unpacked_c.data(), sizeof(unpacked_c)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK((unpacked_a == std::array<float, 4>{12, 11, 10, 9}));
    CHECK((unpacked_b == std::array<float, 4>{8, 7, 6, 5}));
    CHECK((unpacked_c == std::array<float, 4>{4, 3, 2, 1}));
    CHECK(!context.pack_rows({&row_a, &row_b}, &batch).is_ok());
    CHECK(!context.unpack_rows(batch, {&row_a, nullptr, &row_c}).is_ok());

    CudaTensor wrong_shape;
    CHECK(CudaTensor::create_contiguous({256}, DType::Float32, &wrong_shape).is_ok());
    CHECK(!context.copy(output, &wrong_shape).is_ok());
    CHECK(!context.vector_add(lhs, wrong_shape, &output).is_ok());
    CHECK(!context.vector_add(lhs, rhs, nullptr).is_ok());

    CudaTensor matrix_lhs;
    CudaTensor matrix_rhs;
    CudaTensor matrix_output;
    CHECK(CudaTensor::create_contiguous({2, 3}, DType::Float32, &matrix_lhs).is_ok());
    CHECK(CudaTensor::create_contiguous({3, 2}, DType::Float32, &matrix_rhs).is_ok());
    CHECK(CudaTensor::create_contiguous({2, 2}, DType::Float32, &matrix_output).is_ok());
    const std::array<float, 6> matrix_lhs_host{1, 2, 3, 4, 5, 6};
    const std::array<float, 6> matrix_rhs_host{7, 8, 9, 10, 11, 12};
    std::array<float, 4> matrix_output_host{};
    CHECK(context.upload(
        &matrix_lhs, matrix_lhs_host.data(), sizeof(matrix_lhs_host)).is_ok());
    CHECK(context.upload(
        &matrix_rhs, matrix_rhs_host.data(), sizeof(matrix_rhs_host)).is_ok());
    CHECK(context.matmul(matrix_lhs, matrix_rhs, &matrix_output).is_ok());
    CHECK(context.download(
        matrix_output, matrix_output_host.data(), sizeof(matrix_output_host)).is_ok());
    CHECK(context.synchronize().is_ok());
    CHECK((matrix_output_host == std::array<float, 4>{58, 64, 139, 154}));
    CHECK(!context.matmul(matrix_lhs, wrong_shape, &matrix_output).is_ok());

    CudaExecutionContext moved(std::move(context));
    CHECK(!context.valid());
    CHECK(moved.valid());
    CHECK(moved.workspace_capacity_bytes() == 2048);
    CHECK(moved.workspace_allocation_count() == 2);
    CHECK(!context.synchronize().is_ok());
    CHECK(moved.synchronize().is_ok());
    return 0;
}

#undef CHECK
