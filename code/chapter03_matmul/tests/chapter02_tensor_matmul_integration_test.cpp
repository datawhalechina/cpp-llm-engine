// Chapter 02→03：Tensor View 到 MatMul 的集成测试
// 职责：验证共享 Storage View 经公共适配器进入 CPU MatMul。
// 通过条件：连续、offset、transpose 连续化正确，错误契约被拒绝。
// 验证：bash code/chapter03_matmul/scripts/run.sh。
#include "chapter02_tensor_ops/tensor.h"
#include "chapter03_matmul/cpu_backend.h"
#include "chapter03_matmul/tensor_matmul_adapter.h"
#include "cpp_llm_engine/testing/test.h"

#include <string_view>

namespace {

class FailingCpuBackend final : public cpp_llm_engine::core::MatmulBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override {
        return cpp_llm_engine::core::BackendKind::CPU;
    }

    std::string_view name() const override { return "failing-cpu"; }

    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::MatmulRequest&) const override {
        return cpp_llm_engine::core::Status::error("injected backend failure");
    }
};

void fill_lhs(chapter02_tensor_ops::Tensor& tensor) {
    float value = 1.0f;
    for (std::size_t row = 0; row < tensor.rows(); ++row) {
        for (std::size_t col = 0; col < tensor.cols(); ++col) {
            tensor.at(row, col) = value++;
        }
    }
}

chapter02_tensor_ops::Tensor make_rhs() {
    chapter02_tensor_ops::Tensor rhs(3, 2);
    rhs.at(0, 0) = 7.0f;
    rhs.at(0, 1) = 8.0f;
    rhs.at(1, 0) = 9.0f;
    rhs.at(1, 1) = 10.0f;
    rhs.at(2, 0) = 11.0f;
    rhs.at(2, 1) = 12.0f;
    return rhs;
}

}  // namespace

int main() {
    using chapter03_matmul::Matrix;
    using chapter03_matmul::cpu_reference_backend;
    using chapter03_matmul::try_compute_tensor_views;
    using chapter03_matmul::try_compute_tensor_views_into;
    cpp_llm_engine::testing::TestSuite suite("chapter02_tensor_matmul_integration");

    suite.run("tensor views enforce layout and preserve output on failure", [&] {
    chapter02_tensor_ops::Tensor lhs_storage(3, 2);
    fill_lhs(lhs_storage);
    const auto lhs = lhs_storage.reshape(2, 3);
    const auto rhs = make_rhs();
    Matrix output;
    auto status = try_compute_tensor_views(
        lhs.view_ref(), rhs.view_ref(), cpu_reference_backend(), &output);
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, output.rows == 2 && output.cols == 2);
    CPP_LLM_CHECK(suite, output.at(0, 0) == 58.0f);
    CPP_LLM_CHECK(suite, output.at(0, 1) == 64.0f);
    CPP_LLM_CHECK(suite, output.at(1, 0) == 139.0f);
    CPP_LLM_CHECK(suite, output.at(1, 1) == 154.0f);

    chapter02_tensor_ops::Tensor tensor_output(2, 2, -1.0f);
    status = try_compute_tensor_views_into(
        lhs.view_ref(), rhs.view_ref(), cpu_reference_backend(),
        tensor_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, tensor_output.at(0, 0) == 58.0f);
    CPP_LLM_CHECK(suite, tensor_output.at(0, 1) == 64.0f);
    CPP_LLM_CHECK(suite, tensor_output.at(1, 0) == 139.0f);
    CPP_LLM_CHECK(suite, tensor_output.at(1, 1) == 154.0f);

    chapter02_tensor_ops::Tensor bad_output_shape(1, 4, -2.0f);
    status = try_compute_tensor_views_into(
        lhs.view_ref(), rhs.view_ref(), cpu_reference_backend(),
        bad_output_shape.mutable_view_ref());
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "matmul output shape mismatch");
    CPP_LLM_CHECK(suite, bad_output_shape.at(0, 0) == -2.0f);

    chapter02_tensor_ops::Tensor offset_owner(2, 3);
    fill_lhs(offset_owner);
    const auto offset_lhs = offset_owner.slice(1, 1, 0, 3);
    chapter02_tensor_ops::Tensor column_rhs(3, 1);
    column_rhs.at(0, 0) = 1.0f;
    column_rhs.at(1, 0) = 2.0f;
    column_rhs.at(2, 0) = 3.0f;
    status = try_compute_tensor_views(
        offset_lhs.view_ref(), column_rhs.view_ref(),
        cpu_reference_backend(), &output);
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, output.rows == 1 && output.cols == 1);
    CPP_LLM_CHECK(suite, output.at(0, 0) == 32.0f);

    const auto non_contiguous = offset_owner.slice(0, 2, 1, 2);
    chapter02_tensor_ops::Tensor two_by_one(2, 1, 1.0f);
    status = try_compute_tensor_views(
        non_contiguous.view_ref(), two_by_one.view_ref(),
        cpu_reference_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "matmul requires contiguous row-major input");

    const auto materialized_lhs = non_contiguous.contiguous();
    status = try_compute_tensor_views(
        materialized_lhs.view_ref(), two_by_one.view_ref(),
        cpu_reference_backend(), &output);
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, output.rows == 2 && output.cols == 1);
    CPP_LLM_CHECK(suite, output.at(0, 0) == 5.0f);
    CPP_LLM_CHECK(suite, output.at(1, 0) == 11.0f);

    const auto transposed_lhs = lhs_storage.transpose();
    chapter02_tensor_ops::Tensor transpose_rhs(3, 1, 1.0f);
    status = try_compute_tensor_views(
        transposed_lhs.view_ref(), transpose_rhs.view_ref(),
        cpu_reference_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "matmul requires contiguous row-major input");

    const auto materialized_transpose = transposed_lhs.contiguous();
    status = try_compute_tensor_views(
        materialized_transpose.view_ref(), transpose_rhs.view_ref(),
        cpu_reference_backend(), &output);
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, output.rows == 2 && output.cols == 1);
    CPP_LLM_CHECK(suite, output.at(0, 0) == 9.0f);
    CPP_LLM_CHECK(suite, output.at(1, 0) == 12.0f);

    chapter02_tensor_ops::Tensor bad_rhs(2, 2, 1.0f);
    status = try_compute_tensor_views(
        lhs.view_ref(), bad_rhs.view_ref(), cpu_reference_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "matmul shape mismatch");
    CPP_LLM_CHECK(suite, output.rows == 2 && output.cols == 1);
    CPP_LLM_CHECK(suite, output.at(1, 0) == 12.0f);

    auto bad_rank = lhs.view_ref();
    bad_rank.contract.shape.dimensions = {1, 2, 3};
    bad_rank.contract.strides = {6, 3, 1};
    status = try_compute_tensor_views(
        bad_rank, rhs.view_ref(), cpu_reference_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "matmul tensor view rank must be two");

    auto short_storage = lhs.view_ref();
    short_storage.contract.storage_bytes = sizeof(float);
    status = try_compute_tensor_views(
        short_storage, rhs.view_ref(), cpu_reference_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "tensor view exceeds storage bytes");

    auto bad_dtype = lhs.view_ref();
    bad_dtype.contract.dtype = cpp_llm_engine::core::DType::Int8;
    status = try_compute_tensor_views(
        bad_dtype, rhs.view_ref(), cpu_reference_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "matmul tensor view dtype must be Float32");

    auto bad_device = lhs.view_ref();
    bad_device.contract.device = cpp_llm_engine::core::DeviceKind::CUDA;
    status = try_compute_tensor_views(
        bad_device, rhs.view_ref(), cpu_reference_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "matmul tensor view device must be CPU");

    auto null_view = lhs.view_ref();
    null_view.storage_data = nullptr;
    status = try_compute_tensor_views(
        null_view, rhs.view_ref(), cpu_reference_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "tensor view storage data must not be null");

    const FailingCpuBackend failing_backend;
    tensor_output = chapter02_tensor_ops::Tensor(2, 2, -3.0f);
    status = try_compute_tensor_views_into(
        lhs.view_ref(), rhs.view_ref(), failing_backend,
        tensor_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "injected backend failure");
    CPP_LLM_CHECK(suite, tensor_output.at(0, 0) == -3.0f);
    CPP_LLM_CHECK(suite, tensor_output.at(1, 1) == -3.0f);

    status = try_compute_tensor_views(
        lhs.view_ref(), rhs.view_ref(), failing_backend, &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "injected backend failure");
    CPP_LLM_CHECK(suite, output.rows == 2 && output.cols == 1);
    CPP_LLM_CHECK(suite, output.at(1, 0) == 12.0f);
    CPP_LLM_CHECK(suite, !try_compute_tensor_views(
        lhs.view_ref(), rhs.view_ref(), cpu_reference_backend(), nullptr).is_ok());

    });
    return suite.finish();
}
