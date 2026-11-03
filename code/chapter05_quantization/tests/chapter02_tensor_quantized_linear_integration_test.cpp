// Chapter 02->05：公共激活 View 到量化 Linear 的集成测试。
// 职责：验证 rank-1/2 激活、三种量化 batch、布局拒绝和失败保持。
// 通过条件：Q4_K/Q6_K/Q8_0 固定结果正确，错误 contract 不覆盖输出。
// 验证：bash code/chapter05_quantization/scripts/run.sh。
#include "chapter02_tensor_ops/tensor.h"
#include "chapter05_quantization/quantized_linear_view_adapter.h"
#include "cpp_llm_engine/core/quantization_contract.h"
#include "cpp_llm_engine/testing/test.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

std::vector<std::uint8_t> make_unit_q4k_block() {
    std::vector<std::uint8_t> weights(
        cpp_llm_engine::core::kQ4KBlockBytes, 0);
    weights[0] = 0x00;
    weights[1] = 0x3c;  // little-endian half 1.0
    for (std::size_t i = 4; i < 12; ++i) weights[i] = 1;
    for (std::size_t i = 12; i < weights.size(); ++i) weights[i] = 0x11;
    return weights;
}

std::vector<std::uint8_t> make_q6k_block() {
    std::vector<std::uint8_t> weights(
        cpp_llm_engine::core::kQ6KBlockBytes, 0);
    weights[0] = 0x21;
    weights[32] = 0x43;
    weights[128] = 0x39;
    for (std::size_t i = 192; i < 208; ++i) weights[i] = 1;
    weights[208] = 0x00;
    weights[209] = 0x3c;  // little-endian half 1.0
    return weights;
}

std::vector<std::uint8_t> make_q8_0_block() {
    std::vector<std::uint8_t> weights(
        cpp_llm_engine::core::kQ8_0BlockBytes, 0);
    weights[0] = 0x00;
    weights[1] = 0x38;  // little-endian half 0.5
    for (std::size_t i = 0; i < cpp_llm_engine::core::kQ8_0BlockElements; ++i) {
        weights[2 + i] = static_cast<std::uint8_t>(
            static_cast<std::int8_t>(static_cast<int>(i) - 16));
    }
    return weights;
}

bool near(float lhs, float rhs) {
    return std::abs(lhs - rhs) < 1e-3f;
}

}  // namespace

int main() {
    using chapter02_tensor_ops::Tensor;
    using chapter05_quantization::QuantizedLinearBackendKind;
    using chapter05_quantization::QuantizedLinearPolicy;
    using chapter05_quantization::QuantizedMatrixView;
    using chapter05_quantization::compute_quantized_linear_views;

    cpp_llm_engine::testing::TestSuite suite(
        "chapter02_tensor_quantized_linear_integration");
    suite.run("一维和二维激活遵守量化 Linear 契约", [&] {
    const auto packed_weights = make_unit_q4k_block();
    const QuantizedMatrixView weights{
        packed_weights.data(), packed_weights.size(), 1, 256,
        cpp_llm_engine::core::WeightFormat::Q4K};

    Tensor vector_input(std::vector<std::size_t>{256}, 2.0f);
    Tensor vector_output(std::vector<std::size_t>{1}, -1.0f);
    auto status = compute_quantized_linear_views(
        weights, vector_input.view_ref(), vector_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, near(vector_output.at({0}), 512.0f));

    Tensor matrix_input(std::vector<std::size_t>{2, 256}, 2.0f);
    for (std::size_t col = 0; col < 256; ++col) {
        matrix_input.at({1, col}) = 3.0f;
    }
    Tensor matrix_output(std::vector<std::size_t>{2, 1}, -1.0f);
    status = compute_quantized_linear_views(
        weights, matrix_input.view_ref(), matrix_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, near(matrix_output.at({0, 0}), 512.0f));
    CPP_LLM_CHECK(suite, near(matrix_output.at({1, 0}), 768.0f));

    const auto q6k_weights = make_q6k_block();
    const QuantizedMatrixView q6k_matrix{
        q6k_weights.data(), q6k_weights.size(), 1, 256,
        cpp_llm_engine::core::WeightFormat::Q6K};
    Tensor q6k_input(std::vector<std::size_t>{256}, 1.0f);
    Tensor q6k_output(std::vector<std::size_t>{1}, -1.0f);
    status = compute_quantized_linear_views(
        q6k_matrix, q6k_input.view_ref(), q6k_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, near(q6k_output.at({0}), -8086.0f));
    Tensor q6k_batch_input(std::vector<std::size_t>{2, 256}, 1.0f);
    for (std::size_t col = 0; col < 256; ++col) {
        q6k_batch_input.at({1, col}) = 2.0f;
    }
    Tensor q6k_batch_output(std::vector<std::size_t>{2, 1}, -1.0f);
    status = compute_quantized_linear_views(
        q6k_matrix, q6k_batch_input.view_ref(), q6k_batch_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, near(q6k_batch_output.at({0, 0}), -8086.0f));
    CPP_LLM_CHECK(suite, near(q6k_batch_output.at({1, 0}), -16172.0f));

    const auto q8_0_weights = make_q8_0_block();
    const QuantizedMatrixView q8_0_matrix{
        q8_0_weights.data(), q8_0_weights.size(), 1, 32,
        cpp_llm_engine::core::WeightFormat::Q8_0};
    Tensor q8_0_input(std::vector<std::size_t>{32}, 2.0f);
    Tensor q8_0_output(std::vector<std::size_t>{1}, -1.0f);
    status = compute_quantized_linear_views(
        q8_0_matrix, q8_0_input.view_ref(), q8_0_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, near(q8_0_output.at({0}), -16.0f));
    Tensor q8_0_batch_input(std::vector<std::size_t>{2, 32}, 2.0f);
    for (std::size_t col = 0; col < 32; ++col) {
        q8_0_batch_input.at({1, col}) = 3.0f;
    }
    Tensor q8_0_batch_output(std::vector<std::size_t>{2, 1}, -1.0f);
    status = compute_quantized_linear_views(
        q8_0_matrix, q8_0_batch_input.view_ref(), q8_0_batch_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, near(q8_0_batch_output.at({0, 0}), -16.0f));
    CPP_LLM_CHECK(suite, near(q8_0_batch_output.at({1, 0}), -24.0f));

    Tensor layout_owner(std::vector<std::size_t>{2, 257}, 2.0f);
    const auto non_contiguous = layout_owner.slice(1, 0, 256);
    Tensor preserved_output(std::vector<std::size_t>{2, 1}, -2.0f);
    status = compute_quantized_linear_views(
        weights, non_contiguous.view_ref(), preserved_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() ==
           "quantized Linear activation must be contiguous row-major");
    CPP_LLM_CHECK(suite, preserved_output.at({0, 0}) == -2.0f);

    Tensor bad_shape_output(std::vector<std::size_t>{1, 2}, -3.0f);
    status = compute_quantized_linear_views(
        weights, matrix_input.view_ref(), bad_shape_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() ==
           "quantized Linear output shape does not match weights");
    CPP_LLM_CHECK(suite, bad_shape_output.at({0, 0}) == -3.0f);

    auto short_weights = weights;
    --short_weights.bytes;
    status = compute_quantized_linear_views(
        short_weights, matrix_input.view_ref(),
        preserved_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() ==
           "quantized weight byte count does not match shape");
    CPP_LLM_CHECK(suite, preserved_output.at({1, 0}) == -2.0f);

    const QuantizedLinearPolicy unavailable_policy{
        QuantizedLinearBackendKind::Avx2Fused,
        QuantizedLinearBackendKind::ScalarFused,
        QuantizedLinearBackendKind::ScalarFused};
    status = compute_quantized_linear_views(
        weights, matrix_input.view_ref(), preserved_output.mutable_view_ref(),
        unavailable_policy);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "Q4_K AVX2 backend is unavailable");
    CPP_LLM_CHECK(suite, preserved_output.at({0, 0}) == -2.0f);

    });
    return suite.finish();
}
