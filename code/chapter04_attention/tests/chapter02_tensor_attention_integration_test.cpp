// Chapter 02->04：N 维 Tensor 到 Attention 的集成测试
// 职责：验证 3D prefill/decode、4D batch 和显式连续化路径。
// 通过条件：shape 合法时数值正确，布局或 contract 错误在 backend 前失败。
// 验证：bash code/chapter04_attention/scripts/run.sh。
#include "chapter04_attention/attention_backend.h"
#include "chapter04_attention/attention.h"
#include "chapter04_attention/decoder_layer.h"
#include "chapter04_attention/feed_forward.h"
#include "chapter04_attention/layer_norm.h"
#include "chapter04_attention/linear_layer.h"
#include "chapter04_attention/tensor_attention_adapter.h"
#include "cpp_llm_engine/testing/test.h"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <vector>

namespace {

class FailingAttentionBackend final : public cpp_llm_engine::core::AttentionBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override {
        return cpp_llm_engine::core::BackendKind::CPU;
    }

    std::string_view name() const override { return "failing-attention"; }

    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::AttentionRequest&) const override {
        return cpp_llm_engine::core::Status::error("injected attention failure");
    }
};

class RecordingAttentionBackend final : public cpp_llm_engine::core::AttentionBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override {
        return cpp_llm_engine::core::BackendKind::CPU;
    }

    std::string_view name() const override { return "recording-attention"; }

    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::AttentionRequest& request) const override {
        ++call_count;
        last_key_tokens = request.key_tokens;
        last_query_position_offset = request.query_position_offset;
        std::fill_n(
            request.output,
            request.query_tokens * request.heads * request.head_dim,
            0.0F);
        return cpp_llm_engine::core::Status::ok();
    }

    mutable std::size_t call_count = 0;
    mutable std::size_t last_key_tokens = 0;
    mutable std::size_t last_query_position_offset = 0;
};

bool near(float lhs, float rhs) {
    return std::abs(lhs - rhs) < 1e-5f;
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter02_tensor_attention_integration");
    suite.run("tensor views cover decoder components and attention layouts", [&] {
    using chapter02_tensor_ops::Tensor;
    using chapter04_attention::cpu_reference_attention_backend;
    using chapter04_attention::try_compute_tensor_attention;
    using chapter04_attention::try_compute_tensor_attention_into;

    Tensor linear_input(1, 2);
    linear_input.at(0, 0) = 1.0f;
    linear_input.at(0, 1) = 2.0f;
    Tensor linear_output(1, 3, -1.0f);
    const chapter04_attention::LinearLayer linear(2, 3);
    auto linear_status = linear.try_forward(
        linear_input.view_ref(), linear_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, linear_status.is_ok());
    CPP_LLM_CHECK(suite, near(linear_output.at(0, 0), 0.05f));
    CPP_LLM_CHECK(suite, near(linear_output.at(0, 1), 0.10f));
    CPP_LLM_CHECK(suite, near(linear_output.at(0, 2), 0.15f));

    Tensor bad_linear_output(3, 1, -2.0f);
    linear_status = linear.try_forward(
        linear_input.view_ref(), bad_linear_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !linear_status.is_ok());
    CPP_LLM_CHECK(suite, linear_status.message() == "linear output shape mismatch");
    CPP_LLM_CHECK(suite, bad_linear_output.at(0, 0) == -2.0f);

    Tensor norm_output(1, 2, -1.0f);
    const chapter04_attention::LayerNorm norm(2);
    auto norm_status = norm.try_forward(
        linear_input.view_ref(), norm_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, norm_status.is_ok());
    CPP_LLM_CHECK(suite, near(norm_output.at(0, 0), -0.99998f));
    CPP_LLM_CHECK(suite, near(norm_output.at(0, 1), 0.99998f));

    Tensor bad_norm_output(2, 1, -3.0f);
    norm_status = norm.try_forward(
        linear_input.view_ref(), bad_norm_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !norm_status.is_ok());
    CPP_LLM_CHECK(suite, norm_status.message() == "layer norm output shape mismatch");
    CPP_LLM_CHECK(suite, bad_norm_output.at(0, 0) == -3.0f);

    Tensor residual_rhs(1, 2);
    residual_rhs.at(0, 0) = 3.0f;
    residual_rhs.at(0, 1) = 4.0f;
    Tensor residual_output(1, 2, -1.0f);
    auto residual_status = chapter04_attention::try_add_views(
        linear_input.view_ref(), residual_rhs.view_ref(),
        residual_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, residual_status.is_ok());
    CPP_LLM_CHECK(suite, residual_output.at(0, 0) == 4.0f);
    CPP_LLM_CHECK(suite, residual_output.at(0, 1) == 6.0f);

    residual_status = chapter04_attention::try_add_views(
        linear_input.view_ref(), residual_rhs.view_ref(),
        linear_input.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, residual_status.is_ok());
    CPP_LLM_CHECK(suite, linear_input.at(0, 0) == 4.0f);
    CPP_LLM_CHECK(suite, linear_input.at(0, 1) == 6.0f);

    Tensor bad_residual_output(2, 1, -4.0f);
    residual_status = chapter04_attention::try_add_views(
        linear_input.view_ref(), residual_rhs.view_ref(),
        bad_residual_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !residual_status.is_ok());
    CPP_LLM_CHECK(suite, residual_status.message() == "residual add shape mismatch");
    CPP_LLM_CHECK(suite, bad_residual_output.at(0, 0) == -4.0f);

    Tensor activation_input(1, 2);
    activation_input.at(0, 0) = -2.0f;
    activation_input.at(0, 1) = 3.0f;
    Tensor activation_output(1, 2, -1.0f);
    auto activation_status = chapter04_attention::try_apply_activation(
        activation_input.view_ref(), chapter04_attention::ActivationKind::Relu,
        activation_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, activation_status.is_ok());
    CPP_LLM_CHECK(suite, activation_output.at(0, 0) == 0.0f);
    CPP_LLM_CHECK(suite, activation_output.at(0, 1) == 3.0f);

    activation_status = chapter04_attention::try_apply_activation(
        activation_input.view_ref(), chapter04_attention::ActivationKind::Relu,
        activation_input.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, activation_status.is_ok());
    CPP_LLM_CHECK(suite, activation_input.at(0, 0) == 0.0f);
    CPP_LLM_CHECK(suite, activation_input.at(0, 1) == 3.0f);

    Tensor bad_activation_output(2, 1, -5.0f);
    activation_status = chapter04_attention::try_apply_activation(
        activation_input.view_ref(), chapter04_attention::ActivationKind::Relu,
        bad_activation_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !activation_status.is_ok());
    CPP_LLM_CHECK(suite, activation_status.message() == "activation output shape mismatch");
    CPP_LLM_CHECK(suite, bad_activation_output.at(0, 0) == -5.0f);

    Tensor ffn_input(1, 2);
    ffn_input.at(0, 0) = 1.0f;
    ffn_input.at(0, 1) = 2.0f;
    Tensor ffn_output(1, 2, -1.0f);
    const chapter04_attention::FeedForward feed_forward(2, 3);
    auto ffn_status = feed_forward.try_forward(
        ffn_input.view_ref(), ffn_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, ffn_status.is_ok());
    CPP_LLM_CHECK(suite, near(ffn_output.at(0, 0), 0.007f));
    CPP_LLM_CHECK(suite, near(ffn_output.at(0, 1), 0.014f));

    Tensor bad_ffn_output(2, 1, -6.0f);
    ffn_status = feed_forward.try_forward(
        ffn_input.view_ref(), bad_ffn_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !ffn_status.is_ok());
    CPP_LLM_CHECK(suite, ffn_status.message() == "feed forward output shape mismatch");
    CPP_LLM_CHECK(suite, bad_ffn_output.at(0, 0) == -6.0f);

    Tensor mha_input(1, 2);
    mha_input.at(0, 0) = 1.0f;
    mha_input.at(0, 1) = 2.0f;
    Tensor mha_output(1, 2, -1.0f);
    const chapter04_attention::MultiHeadAttention mha(2, 1);
    auto mha_status = mha.try_forward(
        mha_input.view_ref(), mha_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, mha_status.is_ok());
    CPP_LLM_CHECK(suite, near(mha_output.at(0, 0), 0.0025f));
    CPP_LLM_CHECK(suite, near(mha_output.at(0, 1), 0.005f));

    Tensor failed_mha_output(1, 2, -7.0f);
    const FailingAttentionBackend failing_mha_backend;
    const chapter04_attention::MultiHeadAttention failing_mha(
        2, 1, failing_mha_backend);
    mha_status = failing_mha.try_forward(
        mha_input.view_ref(), failed_mha_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !mha_status.is_ok());
    CPP_LLM_CHECK(suite, mha_status.message() == "injected attention failure");
    CPP_LLM_CHECK(suite, failed_mha_output.at(0, 0) == -7.0f);
    CPP_LLM_CHECK(suite, failed_mha_output.at(0, 1) == -7.0f);

    Tensor decoder_input(1, 2);
    decoder_input.at(0, 0) = 1.0f;
    decoder_input.at(0, 1) = 2.0f;
    Tensor decoder_output(1, 2, -1.0f);
    const chapter04_attention::DecoderLayer decoder(2, 1, 3, 0);
    auto decoder_status = decoder.try_forward(
        decoder_input.view_ref(), decoder_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, decoder_status.is_ok());
    chapter04_attention::Tensor teaching_decoder_input(1, 2);
    teaching_decoder_input.at(0, 0) = 1.0f;
    teaching_decoder_input.at(0, 1) = 2.0f;
    const auto teaching_decoder_output = decoder.forward(teaching_decoder_input);
    CPP_LLM_CHECK(suite, teaching_decoder_output.ready());
    CPP_LLM_CHECK(suite, near(decoder_output.at(0, 0), teaching_decoder_output.at(0, 0)));
    CPP_LLM_CHECK(suite, near(decoder_output.at(0, 1), teaching_decoder_output.at(0, 1)));

    Tensor bad_decoder_output(2, 1, -9.0f);
    decoder_status = decoder.try_forward(
        decoder_input.view_ref(), bad_decoder_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !decoder_status.is_ok());
    CPP_LLM_CHECK(suite, decoder_status.message() == "decoder layer output shape mismatch");
    CPP_LLM_CHECK(suite, bad_decoder_output.at(0, 0) == -9.0f);

    Tensor query(std::vector<std::size_t>{2, 1, 1}, 1.0f);
    Tensor key(std::vector<std::size_t>{2, 1, 1}, 1.0f);
    Tensor value(std::vector<std::size_t>{2, 1, 1});
    value.at({0, 0, 0}) = 2.0f;
    value.at({1, 0, 0}) = 4.0f;

    Tensor output;
    auto status = try_compute_tensor_attention(
        query.view_ref(), key.view_ref(), value.view_ref(),
        cpu_reference_attention_backend(), &output);
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, output.shape() == std::vector<std::size_t>({2, 1, 1}));
    CPP_LLM_CHECK(suite, near(output.at({0, 0, 0}), 2.0f));
    CPP_LLM_CHECK(suite, near(output.at({1, 0, 0}), 3.0f));

    Tensor view_output(std::vector<std::size_t>{2, 1, 1}, -1.0f);
    status = try_compute_tensor_attention_into(
        query.view_ref(), key.view_ref(), value.view_ref(),
        cpu_reference_attention_backend(), view_output.mutable_view_ref());
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, near(view_output.at({0, 0, 0}), 2.0f));
    CPP_LLM_CHECK(suite, near(view_output.at({1, 0, 0}), 3.0f));

    Tensor bad_attention_output(std::vector<std::size_t>{1, 2, 1}, -7.0f);
    status = try_compute_tensor_attention_into(
        query.view_ref(), key.view_ref(), value.view_ref(),
        cpu_reference_attention_backend(),
        bad_attention_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "attention output shape must match query");
    CPP_LLM_CHECK(suite, bad_attention_output.at({0, 0, 0}) == -7.0f);

    Tensor failed_attention_output(std::vector<std::size_t>{2, 1, 1}, -8.0f);
    const FailingAttentionBackend failing_attention_backend;
    status = try_compute_tensor_attention_into(
        query.view_ref(), key.view_ref(), value.view_ref(),
        failing_attention_backend, failed_attention_output.mutable_view_ref());
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "injected attention failure");
    CPP_LLM_CHECK(suite, failed_attention_output.at({0, 0, 0}) == -8.0f);
    CPP_LLM_CHECK(suite, failed_attention_output.at({1, 0, 0}) == -8.0f);

    Tensor decode_query(std::vector<std::size_t>{1, 1, 1}, 1.0f);
    value.at({1, 0, 0}) = 100.0f;
    status = try_compute_tensor_attention(
        decode_query.view_ref(), key.view_ref(), value.view_ref(),
        cpu_reference_attention_backend(), &output, 1);
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, near(output.at({0, 0, 0}), 51.0f));

    Tensor batch_query(std::vector<std::size_t>{2, 1, 1, 1}, 1.0f);
    Tensor batch_key(std::vector<std::size_t>{2, 1, 1, 1}, 1.0f);
    Tensor batch_value(std::vector<std::size_t>{2, 1, 1, 1});
    batch_value.at({0, 0, 0, 0}) = 3.0f;
    batch_value.at({1, 0, 0, 0}) = 7.0f;
    status = try_compute_tensor_attention(
        batch_query.view_ref(), batch_key.view_ref(), batch_value.view_ref(),
        cpu_reference_attention_backend(), &output);
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, output.shape() == std::vector<std::size_t>({2, 1, 1, 1}));
    CPP_LLM_CHECK(suite, near(output.at({0, 0, 0, 0}), 3.0f));
    CPP_LLM_CHECK(suite, near(output.at({1, 0, 0, 0}), 7.0f));

    Tensor layout_source(std::vector<std::size_t>{1, 2, 3, 1}, 1.0f);
    const auto non_contiguous_query = layout_source.transpose(1, 2);
    Tensor layout_key(std::vector<std::size_t>{1, 3, 2, 1}, 1.0f);
    Tensor layout_value(std::vector<std::size_t>{1, 3, 2, 1}, 2.0f);
    status = try_compute_tensor_attention(
        non_contiguous_query.view_ref(), layout_key.view_ref(), layout_value.view_ref(),
        cpu_reference_attention_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "attention requires contiguous token-major input");

    const auto contiguous_query = non_contiguous_query.contiguous();
    status = try_compute_tensor_attention(
        contiguous_query.view_ref(), layout_key.view_ref(), layout_value.view_ref(),
        cpu_reference_attention_backend(), &output);
    CPP_LLM_REQUIRE(suite, status.is_ok());
    CPP_LLM_CHECK(suite, output.shape() == std::vector<std::size_t>({1, 3, 2, 1}));
    CPP_LLM_CHECK(suite, near(output.at({0, 2, 1, 0}), 2.0f));

    Tensor bad_value(std::vector<std::size_t>{1, 1, 1});
    status = try_compute_tensor_attention(
        query.view_ref(), key.view_ref(), bad_value.view_ref(),
        cpu_reference_attention_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "attention key and value token counts must match");

    auto bad_dtype = query.view_ref();
    bad_dtype.contract.dtype = cpp_llm_engine::core::DType::Int8;
    status = try_compute_tensor_attention(
        bad_dtype, key.view_ref(), value.view_ref(),
        cpu_reference_attention_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "attention tensor view dtype must be Float32");

    auto bad_device = query.view_ref();
    bad_device.contract.device = cpp_llm_engine::core::DeviceKind::CUDA;
    status = try_compute_tensor_attention(
        bad_device, key.view_ref(), value.view_ref(),
        cpu_reference_attention_backend(), &output);
    CPP_LLM_CHECK(suite, !status.is_ok());
    CPP_LLM_CHECK(suite, status.message() == "attention tensor view device must be CPU");
    CPP_LLM_CHECK(suite, !try_compute_tensor_attention(
        query.view_ref(), key.view_ref(), value.view_ref(),
        cpu_reference_attention_backend(), nullptr).is_ok());

    });
    suite.run("cached attention exposes growing history to the backend", [&] {
        RecordingAttentionBackend backend;
        const chapter04_attention::MultiHeadAttention attention(2, 1, backend);
        chapter04_attention::KvCache cache(3, 1, 2);

        chapter02_tensor_ops::Tensor prefill_input(2, 2, 1.0F);
        chapter02_tensor_ops::Tensor prefill_output(2, 2, -1.0F);
        auto status = attention.try_forward_with_cache(
            prefill_input.view_ref(), 0, &cache,
            prefill_output.mutable_view_ref());
        CPP_LLM_REQUIRE(suite, status.is_ok());
        CPP_LLM_CHECK(suite, cache.size() == 2);
        CPP_LLM_CHECK(suite, backend.call_count == 1);
        CPP_LLM_CHECK(suite, backend.last_key_tokens == 2);
        CPP_LLM_CHECK(suite, backend.last_query_position_offset == 0);

        chapter02_tensor_ops::Tensor decode_input(1, 2, 2.0F);
        chapter02_tensor_ops::Tensor decode_output(1, 2, -2.0F);
        status = attention.try_forward_with_cache(
            decode_input.view_ref(), 2, &cache,
            decode_output.mutable_view_ref());
        CPP_LLM_REQUIRE(suite, status.is_ok());
        CPP_LLM_CHECK(suite, cache.size() == 3);
        CPP_LLM_CHECK(suite, backend.call_count == 2);
        CPP_LLM_CHECK(suite, backend.last_key_tokens == 3);
        CPP_LLM_CHECK(suite, backend.last_query_position_offset == 2);

        chapter02_tensor_ops::Tensor failed_output(1, 2, -3.0F);
        status = attention.try_forward_with_cache(
            decode_input.view_ref(), 3, &cache,
            failed_output.mutable_view_ref());
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, cache.size() == 3);
        CPP_LLM_CHECK(suite, backend.call_count == 2);
        CPP_LLM_CHECK(suite, failed_output.at(0, 0) == -3.0F);

        const FailingAttentionBackend failing_backend;
        const chapter04_attention::MultiHeadAttention failing_attention(
            2, 1, failing_backend);
        chapter04_attention::KvCache unchanged_cache(2, 1, 2);
        chapter02_tensor_ops::Tensor backend_failed_output(2, 2, -4.0F);
        status = failing_attention.try_forward_with_cache(
            prefill_input.view_ref(), 0, &unchanged_cache,
            backend_failed_output.mutable_view_ref());
        CPP_LLM_CHECK(suite, !status.is_ok());
        CPP_LLM_CHECK(suite, unchanged_cache.size() == 0);
        CPP_LLM_CHECK(suite, backend_failed_output.at(0, 0) == -4.0F);
    });
    return suite.finish();
}
