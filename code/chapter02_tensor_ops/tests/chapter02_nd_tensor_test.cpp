// Chapter 02：N 维 Tensor 命名契约测试。
// 职责：验证 1-4 维索引、stride、轴切片、轴交换和连续化。
// 验证入口：Chapter02 CTest 或带 contract、tensor 标签的 CTest。
#include "chapter02_tensor_ops/tensor.h"

#include "cpp_llm_engine/testing/test.h"

#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using cpp_llm_engine::testing::TestSuite;

template <typename Exception, typename Function>
void expect_exception(TestSuite& suite, Function&& function) {
    bool threw = false;
    try {
        function();
    } catch (const Exception&) {
        threw = true;
    }
    CPP_LLM_CHECK(suite, threw);
}

void fill_3d(chapter02_tensor_ops::Tensor& tensor) {
    float value = 0.0f;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                tensor.at({i, j, k}) = value++;
            }
        }
    }
}

}  // namespace

int main() {
    using chapter02_tensor_ops::Tensor;
    TestSuite suite("chapter02_nd_tensor");

    suite.run("one and three dimensional layout", [&] {
        Tensor vector({4});
        CPP_LLM_CHECK(suite, vector.rank() == 1);
        CPP_LLM_CHECK(suite, vector.shape() == std::vector<std::size_t>({4}));
        CPP_LLM_CHECK(suite, vector.strides() == std::vector<std::size_t>({1}));
        CPP_LLM_CHECK(suite, vector.numel() == 4);
        vector.at({3}) = 7.0f;
        CPP_LLM_CHECK(suite, vector.at({3}) == 7.0f);

        Tensor tensor3d(std::vector<std::size_t>{2, 3, 4});
        fill_3d(tensor3d);
        CPP_LLM_CHECK(suite, tensor3d.rank() == 3);
        CPP_LLM_CHECK(
            suite, tensor3d.strides() == std::vector<std::size_t>({12, 4, 1}));
        CPP_LLM_CHECK(suite, tensor3d.at({1, 2, 3}) == 23.0f);
        CPP_LLM_CHECK(suite, tensor3d.data() != nullptr);
        CPP_LLM_CHECK(suite, tensor3d.data()[23] == 23.0f);
        CPP_LLM_CHECK(suite, tensor3d.contract().shape.dimensions == tensor3d.shape());
        CPP_LLM_CHECK(
            suite, tensor3d.contract().validate_element_view(sizeof(float)).is_ok());
    });

    suite.run("axis views share storage and preserve addressing", [&] {
        Tensor tensor3d(std::vector<std::size_t>{2, 3, 4});
        fill_3d(tensor3d);

        const auto axis_slice = tensor3d.slice(1, 1, 2);
        CPP_LLM_CHECK(
            suite, axis_slice.shape() == std::vector<std::size_t>({2, 2, 4}));
        CPP_LLM_CHECK(
            suite, axis_slice.strides() == std::vector<std::size_t>({12, 4, 1}));
        CPP_LLM_CHECK(suite, axis_slice.offset() == 4);
        CPP_LLM_CHECK(suite, axis_slice.shares_storage_with(tensor3d));
        CPP_LLM_CHECK(suite, axis_slice.at({1, 1, 3}) == 23.0f);

        const auto transposed = tensor3d.transpose(0, 2);
        CPP_LLM_CHECK(
            suite, transposed.shape() == std::vector<std::size_t>({4, 3, 2}));
        CPP_LLM_CHECK(
            suite, transposed.strides() == std::vector<std::size_t>({1, 4, 12}));
        CPP_LLM_CHECK(suite, !transposed.is_contiguous());
        CPP_LLM_CHECK(suite, transposed.data() == nullptr);
        CPP_LLM_CHECK(suite, transposed.shares_storage_with(tensor3d));
        CPP_LLM_CHECK(suite, transposed.at({3, 2, 1}) == 23.0f);
    });

    suite.run("materialize and reshape preserve values", [&] {
        Tensor tensor3d(std::vector<std::size_t>{2, 3, 4});
        fill_3d(tensor3d);
        const auto transposed = tensor3d.transpose(0, 2);
        const auto materialized = transposed.contiguous();
        CPP_LLM_CHECK(suite, materialized.is_contiguous());
        CPP_LLM_CHECK(suite, materialized.data() != nullptr);
        CPP_LLM_CHECK(
            suite, materialized.strides() == std::vector<std::size_t>({6, 2, 1}));
        CPP_LLM_CHECK(suite, !materialized.shares_storage_with(transposed));
        CPP_LLM_CHECK(suite, materialized.at({3, 2, 1}) == 23.0f);

        const auto reshaped = tensor3d.reshape({2, 2, 2, 3});
        CPP_LLM_CHECK(suite, reshaped.rank() == 4);
        CPP_LLM_CHECK(
            suite, reshaped.shape() == std::vector<std::size_t>({2, 2, 2, 3}));
        CPP_LLM_CHECK(
            suite, reshaped.strides() == std::vector<std::size_t>({12, 6, 3, 1}));
        CPP_LLM_CHECK(suite, reshaped.shares_storage_with(tensor3d));
        CPP_LLM_CHECK(suite, reshaped.at({1, 0, 1, 2}) == 17.0f);

        const auto transposed4d = reshaped.transpose(1, 2);
        CPP_LLM_CHECK(
            suite, transposed4d.strides() == std::vector<std::size_t>({12, 3, 6, 1}));
        CPP_LLM_CHECK(suite, !transposed4d.is_contiguous());
        CPP_LLM_CHECK(suite, transposed4d.at({1, 1, 0, 2}) == 17.0f);
        const auto contiguous4d = transposed4d.contiguous();
        CPP_LLM_CHECK(
            suite, contiguous4d.strides() == std::vector<std::size_t>({12, 6, 3, 1}));
        CPP_LLM_CHECK(suite, contiguous4d.at({1, 1, 0, 2}) == 17.0f);
    });

    suite.run("in-place transforms are atomic", [&] {
        Tensor tensor3d(std::vector<std::size_t>{2, 3, 4});
        fill_3d(tensor3d);

        Tensor in_place_reshape = tensor3d;
        auto status = in_place_reshape.try_reshape({4, 6}, &in_place_reshape);
        CPP_LLM_CHECK(suite, status.is_ok());
        CPP_LLM_CHECK(
            suite, in_place_reshape.shape() == std::vector<std::size_t>({4, 6}));
        CPP_LLM_CHECK(suite, in_place_reshape.at({3, 5}) == 23.0f);

        Tensor in_place_transpose = tensor3d;
        status = in_place_transpose.try_transpose(0, 2, &in_place_transpose);
        CPP_LLM_CHECK(suite, status.is_ok());
        CPP_LLM_CHECK(
            suite, in_place_transpose.shape() == std::vector<std::size_t>({4, 3, 2}));
        CPP_LLM_CHECK(suite, in_place_transpose.at({3, 2, 1}) == 23.0f);
        status = in_place_transpose.try_slice(0, 1, 2, &in_place_transpose);
        CPP_LLM_CHECK(suite, status.is_ok());
        CPP_LLM_CHECK(
            suite, in_place_transpose.shape() == std::vector<std::size_t>({2, 3, 2}));
        CPP_LLM_CHECK(suite, in_place_transpose.offset() == 1);
        CPP_LLM_CHECK(suite, in_place_transpose.at({0, 2, 1}) == 21.0f);

        const auto preserved_shape = in_place_transpose.shape();
        const auto preserved_offset = in_place_transpose.offset();
        status = in_place_transpose.try_slice(3, 0, 1, &in_place_transpose);
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, in_place_transpose.shape() == preserved_shape);
        CPP_LLM_CHECK(suite, in_place_transpose.offset() == preserved_offset);
        CPP_LLM_CHECK(suite, in_place_transpose.at({0, 2, 1}) == 21.0f);

        Tensor in_place_contiguous = tensor3d.transpose(0, 2);
        const Tensor source_view = in_place_contiguous;
        CPP_LLM_CHECK(
            suite, in_place_contiguous.try_contiguous(&in_place_contiguous).is_ok());
        CPP_LLM_CHECK(suite, in_place_contiguous.is_contiguous());
        CPP_LLM_CHECK(suite, !in_place_contiguous.shares_storage_with(source_view));
        CPP_LLM_CHECK(suite, in_place_contiguous.at({3, 2, 1}) == 23.0f);
    });

    suite.run("invalid rank axis and shape are rejected", [&] {
        Tensor tensor3d(std::vector<std::size_t>{2, 3, 4});
        fill_3d(tensor3d);
        Tensor output;
        CPP_LLM_CHECK(suite, !tensor3d.try_slice(3, 0, 1, &output).is_ok());
        CPP_LLM_CHECK(suite, !tensor3d.try_slice(1, 3, 1, &output).is_ok());
        CPP_LLM_CHECK(suite, !tensor3d.try_transpose(0, 3, &output).is_ok());
        CPP_LLM_CHECK(suite, !tensor3d.try_reshape({2, 3, 5}, &output).is_ok());
        CPP_LLM_CHECK(suite, !tensor3d.try_reshape(
            {std::numeric_limits<std::size_t>::max(), 2}, &output).is_ok());
        CPP_LLM_CHECK(suite, !tensor3d.try_reshape({2, 3, 4}, nullptr).is_ok());
        expect_exception<std::out_of_range>(suite, [&] { tensor3d.at({0, 0}); });
        expect_exception<std::out_of_range>(suite, [&] { tensor3d.at({2, 0, 0}); });
        expect_exception<std::out_of_range>(suite, [&] { tensor3d.rows(); });
        expect_exception<std::length_error>(suite, [] {
            Tensor invalid(std::vector<std::size_t>{});
        });
        expect_exception<std::length_error>(suite, [] {
            Tensor invalid(std::vector<std::size_t>{1, 1, 1, 1, 1});
        });
        expect_exception<std::length_error>(suite, [] {
            Tensor invalid(std::vector<std::size_t>{2, 0, 3});
        });
        expect_exception<std::length_error>(suite, [] {
            Tensor invalid(std::vector<std::size_t>{
                std::numeric_limits<std::size_t>::max(), 2});
        });
    });

    return suite.finish();
}
