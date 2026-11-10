// Model：验证三维 packed expert 到二维 Linear View 的布局契约。
// 职责：覆盖 K-quant/Q5 偏移、格式映射和失败原子性。
// 通过条件：每个 expert 指向独立连续窗口，错误输入保留旧 View。
#include "cpp_llm_engine/model/packed_expert_view.h"
#include "cpp_llm_engine/testing/test.h"

#include <cstddef>
#include <vector>

int main() {
    cpp_llm_engine::testing::TestSuite suite("packed_expert_view");
    using cpp_llm_engine::model::PackedExpertView;

    suite.run("Q3_K expert dimension selects a contiguous matrix", [&] {
        std::vector<std::byte> bytes(3 * 2 * 110, std::byte{0});
        bytes[0] = std::byte{1};
        bytes[220] = std::byte{2};
        bytes[440] = std::byte{3};
        const auto tensor = chapter01::Tensor::from_raw_bytes(
            "experts", {256, 2, 3}, chapter01::DType::Q3_K, std::move(bytes));
        PackedExpertView view;
        std::string error;
        CPP_LLM_REQUIRE(suite, PackedExpertView::create(tensor, 1, &view, &error));
        CPP_LLM_CHECK(suite, view.rows == 2 && view.cols == 256 && view.bytes == 220);
        CPP_LLM_CHECK(suite, view.format == cpp_llm_engine::core::WeightFormat::Q3K);
        CPP_LLM_CHECK(suite, view.data[0] == std::uint8_t{2});
    });

    suite.run("Q5_1 validates index and preserves output on failure", [&] {
        std::vector<std::byte> bytes(2 * 2 * 24, std::byte{0x5});
        const auto tensor = chapter01::Tensor::from_raw_bytes(
            "experts", {32, 2, 2}, chapter01::DType::Q5_1, std::move(bytes));
        PackedExpertView view;
        CPP_LLM_REQUIRE(suite, PackedExpertView::create(tensor, 0, &view));
        const auto* preserved = view.data;
        std::string error;
        CPP_LLM_CHECK(suite, !PackedExpertView::create(tensor, 2, &view, &error));
        CPP_LLM_CHECK(suite, view.data == preserved);
        CPP_LLM_CHECK(suite, error == "expert index is outside packed tensor");
    });

    suite.run("Q5_0 maps each expert to a 22-byte block", [&] {
        std::vector<std::byte> bytes(2 * 22, std::byte{0});
        bytes[22] = std::byte{9};
        const auto tensor = chapter01::Tensor::from_raw_bytes(
            "experts", {32, 1, 2}, chapter01::DType::Q5_0,
            std::move(bytes));
        PackedExpertView view;
        CPP_LLM_REQUIRE(suite, PackedExpertView::create(tensor, 1, &view));
        CPP_LLM_CHECK(suite, view.bytes == 22);
        CPP_LLM_CHECK(
            suite, view.format == cpp_llm_engine::core::WeightFormat::Q5_0);
        CPP_LLM_CHECK(suite, view.data[0] == std::uint8_t{9});
    });

    suite.run("DeepSeek expert formats preserve block boundaries", [&] {
        std::vector<std::byte> q2_bytes(2 * 84, std::byte{0});
        q2_bytes[84] = std::byte{0x2a};
        const auto q2 = chapter01::Tensor::from_raw_bytes(
            "q2-experts", {256, 1, 2}, chapter01::DType::Q2_K,
            std::move(q2_bytes));
        PackedExpertView q2_view;
        CPP_LLM_REQUIRE(suite, PackedExpertView::create(q2, 1, &q2_view));
        CPP_LLM_CHECK(suite, q2_view.bytes == 84);
        CPP_LLM_CHECK(
            suite, q2_view.format == cpp_llm_engine::core::WeightFormat::Q2K);
        CPP_LLM_CHECK(suite, q2_view.data[0] == std::uint8_t{0x2a});

        std::vector<std::byte> iq4_bytes(2 * 18, std::byte{0});
        iq4_bytes[18] = std::byte{0x4e};
        const auto iq4 = chapter01::Tensor::from_raw_bytes(
            "iq4-experts", {32, 1, 2}, chapter01::DType::IQ4_NL,
            std::move(iq4_bytes));
        PackedExpertView iq4_view;
        CPP_LLM_REQUIRE(suite, PackedExpertView::create(iq4, 1, &iq4_view));
        CPP_LLM_CHECK(suite, iq4_view.bytes == 18);
        CPP_LLM_CHECK(
            suite,
            iq4_view.format == cpp_llm_engine::core::WeightFormat::IQ4_NL);
        CPP_LLM_CHECK(suite, iq4_view.data[0] == std::uint8_t{0x4e});
    });
    return suite.finish();
}
