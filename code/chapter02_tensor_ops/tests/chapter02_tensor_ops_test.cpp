// Chapter 02：二维 Tensor View 命名契约测试。
// 职责：验证 reshape、slice、transpose、连续化、失败路径和生命周期。
// 验证入口：Chapter02 CTest 或带 unit、tensor 标签的 CTest。
#include "chapter02_tensor_ops/tensor.h"

#include "cpp_llm_engine/testing/test.h"

#include <stdexcept>
#include <vector>

namespace {

using chapter02_tensor_ops::Tensor;
using cpp_llm_engine::testing::TestSuite;

Tensor make_tensor() {
    Tensor tensor(2, 3);
    float value = 1.0f;
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            tensor.at(row, col) = value++;
        }
    }
    return tensor;
}

template <typename Function>
void expect_out_of_range(TestSuite& suite, Function&& function) {
    bool threw = false;
    try {
        function();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CPP_LLM_CHECK(suite, threw);
}

}  // namespace

int main() {
    TestSuite suite("chapter02_tensor_views");

    suite.run("base tensor exposes a valid contiguous contract", [&] {
        Tensor tensor = make_tensor();
        CPP_LLM_CHECK(suite, tensor.ready());
        CPP_LLM_CHECK(suite, tensor.is_contiguous());
        const auto contract = tensor.contract();
        CPP_LLM_CHECK(
            suite, contract.shape.dimensions == std::vector<std::size_t>({2, 3}));
        CPP_LLM_CHECK(
            suite, contract.strides == std::vector<std::size_t>({3, 1}));
        CPP_LLM_CHECK(suite, contract.storage_offset_elements == 0);
        CPP_LLM_CHECK(suite, contract.validate_element_view(sizeof(float)).is_ok());
        const auto view = tensor.view_ref();
        CPP_LLM_CHECK(suite, view.storage_data != nullptr);
        CPP_LLM_CHECK(suite, view.validate(sizeof(float)).is_ok());
        auto mutable_view = tensor.mutable_view_ref();
        CPP_LLM_CHECK(suite, mutable_view.storage_data == view.storage_data);
        CPP_LLM_CHECK(suite, mutable_view.validate(sizeof(float)).is_ok());
        CPP_LLM_CHECK(suite, mutable_view.as_const().storage_data == view.storage_data);
    });

    suite.run("reshape reuses contiguous storage", [&] {
        Tensor tensor = make_tensor();
        const Tensor reshaped = tensor.reshape(3, 2);
        CPP_LLM_CHECK(suite, reshaped.ready());
        CPP_LLM_CHECK(suite, reshaped.rows() == 3);
        CPP_LLM_CHECK(suite, reshaped.cols() == 2);
        CPP_LLM_CHECK(suite, reshaped.offset() == 0);
        CPP_LLM_CHECK(suite, reshaped.row_stride() == 2);
        CPP_LLM_CHECK(suite, reshaped.col_stride() == 1);
        CPP_LLM_CHECK(suite, reshaped.shares_storage_with(tensor));
        CPP_LLM_CHECK(suite, reshaped.at(0, 0) == 1.0f);
        CPP_LLM_CHECK(suite, reshaped.at(0, 1) == 2.0f);
        CPP_LLM_CHECK(suite, reshaped.at(1, 0) == 3.0f);
        CPP_LLM_CHECK(suite, reshaped.at(2, 1) == 6.0f);
    });

    suite.run("slice shares storage and materialize isolates writes", [&] {
        Tensor tensor = make_tensor();
        Tensor sliced = tensor.slice(0, 2, 1, 2);
        CPP_LLM_CHECK(suite, sliced.ready());
        CPP_LLM_CHECK(suite, sliced.rows() == 2);
        CPP_LLM_CHECK(suite, sliced.cols() == 2);
        CPP_LLM_CHECK(suite, sliced.offset() == 1);
        CPP_LLM_CHECK(suite, sliced.row_stride() == 3);
        CPP_LLM_CHECK(suite, sliced.col_stride() == 1);
        CPP_LLM_CHECK(suite, !sliced.is_contiguous());
        CPP_LLM_CHECK(suite, sliced.shares_storage_with(tensor));
        const auto sliced_contract = sliced.contract();
        CPP_LLM_CHECK(
            suite, sliced_contract.shape.dimensions == std::vector<std::size_t>({2, 2}));
        CPP_LLM_CHECK(
            suite, sliced_contract.strides == std::vector<std::size_t>({3, 1}));
        CPP_LLM_CHECK(suite, sliced_contract.storage_offset_elements == 1);
        CPP_LLM_CHECK(
            suite, sliced_contract.validate_element_view(sizeof(float)).is_ok());
        const auto tensor_view = tensor.view_ref();
        const auto sliced_view = sliced.view_ref();
        CPP_LLM_CHECK(suite, sliced_view.storage_data == tensor_view.storage_data);
        CPP_LLM_CHECK(suite, sliced_view.contract.storage_offset_elements == 1);
        CPP_LLM_CHECK(
            suite, sliced_view.contract.strides == std::vector<std::size_t>({3, 1}));
        CPP_LLM_CHECK(suite, sliced.at(0, 0) == 2.0f);
        CPP_LLM_CHECK(suite, sliced.at(1, 1) == 6.0f);

        auto mutable_view = sliced.mutable_view_ref();
        CPP_LLM_CHECK(suite, mutable_view.validate(sizeof(float)).is_ok());
        static_cast<float*>(mutable_view.storage_data)[
            mutable_view.contract.storage_offset_elements] = 41.0f;
        CPP_LLM_CHECK(suite, tensor.at(0, 1) == 41.0f);
        sliced.at(0, 0) = 42.0f;
        CPP_LLM_CHECK(suite, tensor.at(0, 1) == 42.0f);

        const Tensor materialized = sliced.contiguous();
        CPP_LLM_CHECK(suite, materialized.ready());
        CPP_LLM_CHECK(suite, materialized.is_contiguous());
        CPP_LLM_CHECK(suite, !materialized.shares_storage_with(sliced));
        CPP_LLM_CHECK(suite, materialized.offset() == 0);
        CPP_LLM_CHECK(suite, materialized.row_stride() == 2);
        CPP_LLM_CHECK(suite, materialized.col_stride() == 1);
        CPP_LLM_CHECK(suite, materialized.at(0, 0) == 42.0f);
        CPP_LLM_CHECK(suite, materialized.at(0, 1) == 3.0f);
        CPP_LLM_CHECK(suite, materialized.at(1, 0) == 5.0f);
        CPP_LLM_CHECK(suite, materialized.at(1, 1) == 6.0f);
        sliced.at(0, 1) = 99.0f;
        CPP_LLM_CHECK(suite, materialized.at(0, 1) == 3.0f);
        const auto flattened = materialized.reshape(1, 4);
        CPP_LLM_CHECK(suite, flattened.ready());
        CPP_LLM_CHECK(suite, flattened.at(0, 2) == 5.0f);
    });

    suite.run("transpose and offset views preserve addressing", [&] {
        Tensor tensor = make_tensor();
        auto transposed = tensor.transpose();
        CPP_LLM_CHECK(suite, transposed.ready());
        CPP_LLM_CHECK(suite, transposed.rows() == 3);
        CPP_LLM_CHECK(suite, transposed.cols() == 2);
        CPP_LLM_CHECK(suite, transposed.row_stride() == 1);
        CPP_LLM_CHECK(suite, transposed.col_stride() == 3);
        CPP_LLM_CHECK(suite, transposed.offset() == 0);
        CPP_LLM_CHECK(suite, !transposed.is_contiguous());
        CPP_LLM_CHECK(suite, transposed.shares_storage_with(tensor));
        CPP_LLM_CHECK(suite, transposed.at(0, 1) == 4.0f);
        CPP_LLM_CHECK(suite, transposed.at(1, 0) == 2.0f);
        transposed.at(2, 1) = 60.0f;
        CPP_LLM_CHECK(suite, tensor.at(1, 2) == 60.0f);

        const auto contiguous = transposed.contiguous();
        CPP_LLM_CHECK(suite, contiguous.is_contiguous());
        CPP_LLM_CHECK(suite, !contiguous.shares_storage_with(transposed));
        CPP_LLM_CHECK(suite, contiguous.row_stride() == 2);
        CPP_LLM_CHECK(suite, contiguous.col_stride() == 1);
        CPP_LLM_CHECK(suite, contiguous.at(0, 0) == 1.0f);
        CPP_LLM_CHECK(suite, contiguous.at(0, 1) == 4.0f);
        CPP_LLM_CHECK(suite, contiguous.at(2, 1) == 60.0f);

        const auto row_vector = tensor.slice(1, 1, 0, 3);
        const auto column_vector = row_vector.transpose();
        CPP_LLM_CHECK(suite, column_vector.rows() == 3);
        CPP_LLM_CHECK(suite, column_vector.cols() == 1);
        CPP_LLM_CHECK(suite, column_vector.is_contiguous());

        const Tensor offset_view = tensor.slice(1, 1, 0, 3);
        CPP_LLM_CHECK(suite, offset_view.is_contiguous());
        const auto reused = offset_view.contiguous();
        CPP_LLM_CHECK(suite, reused.shares_storage_with(offset_view));
        CPP_LLM_CHECK(suite, reused.offset() == 3);
        const auto offset_ref = offset_view.view_ref();
        CPP_LLM_CHECK(suite, offset_ref.storage_data == tensor.view_ref().storage_data);
        CPP_LLM_CHECK(suite, offset_ref.contract.storage_offset_elements == 3);
        const Tensor offset_reshaped = offset_view.reshape(3, 1);
        CPP_LLM_CHECK(suite, offset_reshaped.ready());
        CPP_LLM_CHECK(suite, offset_reshaped.offset() == 3);
        CPP_LLM_CHECK(suite, offset_reshaped.at(0, 0) == 4.0f);
        CPP_LLM_CHECK(suite, offset_reshaped.at(2, 0) == 60.0f);
    });

    suite.run("invalid transforms fail without a usable output", [&] {
        Tensor tensor = make_tensor();
        const Tensor sliced = tensor.slice(0, 2, 1, 2);
        CPP_LLM_CHECK(suite, !sliced.reshape(4, 1).ready());

        Tensor output;
        const auto non_contiguous = sliced.try_reshape(4, 1, &output);
        CPP_LLM_CHECK(suite, !non_contiguous.is_ok());
        CPP_LLM_CHECK(
            suite, non_contiguous.message() == "reshape requires a contiguous tensor");
        const auto wrong_shape = tensor.try_reshape(4, 2, &output);
        CPP_LLM_CHECK(suite, !wrong_shape.is_ok());
        CPP_LLM_CHECK(
            suite, wrong_shape.message() == "reshape element count must not change");
        CPP_LLM_CHECK(suite, !tensor.try_reshape(3, 2, nullptr).is_ok());
        CPP_LLM_CHECK(suite, !tensor.try_contiguous(nullptr).is_ok());
        CPP_LLM_CHECK(suite, !tensor.try_transpose(nullptr).is_ok());

        Tensor empty;
        CPP_LLM_CHECK(suite, !empty.mutable_view_ref().validate(sizeof(float)).is_ok());
        CPP_LLM_CHECK(suite, !empty.try_contiguous(&output).is_ok());
        CPP_LLM_CHECK(suite, !empty.try_transpose(&output).is_ok());
        CPP_LLM_CHECK(suite, !output.ready());
        CPP_LLM_CHECK(suite, !tensor.slice(2, 1, 0, 1).ready());
        CPP_LLM_CHECK(suite, !tensor.slice(0, 1, 3, 1).ready());
        CPP_LLM_CHECK(suite, !tensor.slice(0, 0, 0, 1).ready());
        const auto bad_slice = tensor.try_slice(1, 2, 0, 1, &output);
        CPP_LLM_CHECK(suite, !bad_slice.is_ok());
        CPP_LLM_CHECK(suite, bad_slice.message() == "slice range exceeds tensor shape");
        CPP_LLM_CHECK(suite, !tensor.try_slice(0, 1, 0, 1, nullptr).is_ok());
        expect_out_of_range(suite, [&] { tensor.at(2, 0); });
        expect_out_of_range(suite, [&] { tensor.at(0, 3); });
    });

    suite.run("view keeps storage alive after owner destruction", [&] {
        Tensor retained_view;
        {
            Tensor owner(1, 2);
            owner.at(0, 0) = 7.0f;
            retained_view = owner.slice(0, 1, 0, 2);
            CPP_LLM_CHECK(suite, owner.storage_use_count() == 2);
            CPP_LLM_CHECK(suite, retained_view.storage_use_count() == 2);
        }
        CPP_LLM_CHECK(suite, retained_view.ready());
        CPP_LLM_CHECK(suite, retained_view.at(0, 0) == 7.0f);
        CPP_LLM_CHECK(suite, retained_view.storage_use_count() == 1);
    });

    return suite.finish();
}
