// Engine CUDA：验证真实 Llama/Qwen3 多层 CPU/CUDA 前向。
// 职责：复用 CPU Model 生成 embedding 与 golden，核对指定层数的 hidden 和分层 K/V。
// 通过条件：两个独立 Context 均提交一个位置，全部层数值保持在量化 CUDA 容差内。
#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"
#include "cpp_llm_engine/model/real_llama_cuda_model.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

namespace {

bool close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float* max_error) {
    if (actual.size() != expected.size()) return false;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);
        maximum = std::max(maximum, error);
        if (!std::isfinite(actual[index]) ||
            error > 6.0e-2F + 8.0e-3F * std::fabs(expected[index])) {
            if (max_error != nullptr) *max_error = maximum;
            return false;
        }
    }
    if (max_error != nullptr) *max_error = maximum;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: real_tinyllama_cuda_multilayer_test <dense-model.gguf> <layers>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);

    constexpr chapter01::TokenId kTokenId = 15043;
    const std::string model_path = argv[1];
    std::size_t layer_count = 0;
    try {
        layer_count = static_cast<std::size_t>(std::stoul(argv[2]));
    } catch (const std::exception&) {
        std::cerr << "layers must be a positive integer\n";
        return 2;
    }
    CHECK(layer_count > 0);
    std::string error;
    auto cpu_model = chapter01::RealLlamaCpuModel::load(
        model_path, layer_count, &error);
    auto cuda_model = cpp_llm_engine::model::RealLlamaCudaModel::load(
        model_path, layer_count, &error);
    CHECK(cpu_model != nullptr && cuda_model != nullptr);
    CHECK(cpu_model->layer_count() == layer_count &&
        cuda_model->layer_count() == layer_count);

    cpp_llm_engine::core::DecoderEmbeddingSequence embeddings;
    CHECK(cpu_model->embed_tokens({kTokenId}, &embeddings, &error));
    CHECK(embeddings.token_count == 1);
    auto cpu_context = cpu_model->create_context();
    auto cuda_context = cuda_model->create_context(1, &error);
    CHECK(cpu_context != nullptr && cuda_context != nullptr);

    std::vector<float> cpu_hidden;
    std::vector<float> cuda_hidden;
    CHECK(cpu_context->forward_hidden_state(
        embeddings.values, &cpu_hidden, &error));
    CHECK(cuda_context->forward_hidden(
        embeddings.values, &cuda_hidden).is_ok());
    CHECK(cpu_context->token_count() == 1 && cuda_context->position() == 1);

    float hidden_error = 0.0F;
    CHECK(close(cuda_hidden, cpu_hidden, &hidden_error));
    float maximum_key_error = 0.0F;
    float maximum_value_error = 0.0F;
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        std::vector<float> cpu_key;
        std::vector<float> cpu_value;
        std::vector<float> cuda_key;
        std::vector<float> cuda_value;
        CHECK(cpu_context->copy_kv_token(
            layer, 0, &cpu_key, &cpu_value, &error));
        CHECK(cuda_context->copy_layer_kv(
            layer, &cuda_key, &cuda_value).is_ok());
        float key_error = 0.0F;
        float value_error = 0.0F;
        CHECK(close(cuda_key, cpu_key, &key_error));
        CHECK(close(cuda_value, cpu_value, &value_error));
        maximum_key_error = std::max(maximum_key_error, key_error);
        maximum_value_error = std::max(maximum_value_error, value_error);
    }

    auto cpu_token_context = cpu_model->create_context();
    auto cuda_token_context = cuda_model->create_context(1, &error);
    CHECK(cpu_token_context != nullptr && cuda_token_context != nullptr);
    std::vector<float> cpu_logits;
    std::vector<float> cuda_logits;
    CHECK(cpu_token_context->prefill({kTokenId}, &cpu_logits, &error));
    CHECK(cuda_token_context->forward_token(kTokenId, &cuda_logits).is_ok());
    CHECK(cpu_logits.size() == cpu_model->vocab_size());
    CHECK(cuda_logits.size() == cuda_model->vocab_size());
    float logits_error = 0.0F;
    CHECK(close(cuda_logits, cpu_logits, &logits_error));
    const auto cpu_greedy = std::max_element(cpu_logits.begin(), cpu_logits.end());
    const auto cuda_greedy = std::max_element(cuda_logits.begin(), cuda_logits.end());
    CHECK(std::distance(cpu_logits.begin(), cpu_greedy) ==
        std::distance(cuda_logits.begin(), cuda_greedy));
    CHECK(!cuda_token_context->forward_token(kTokenId, &cuda_logits).is_ok());
    CHECK(cuda_token_context->position() == 1);

    auto three_stage_context = cuda_model->create_context(2, &error);
    auto fused_context = cuda_model->create_context(2, &error);
    CHECK(three_stage_context != nullptr && fused_context != nullptr);
    fused_context->set_attention_path(
        cpp_llm_engine::cuda::CudaAttentionPath::FusedOnline);
    float fused_hidden_error = 0.0F;
    float fused_logits_error = 0.0F;
    for (std::size_t step = 0; step < 2; ++step) {
        std::vector<float> three_stage_hidden;
        std::vector<float> fused_hidden;
        CHECK(three_stage_context->forward_hidden(
            embeddings.values, &three_stage_hidden).is_ok());
        CHECK(fused_context->forward_hidden(
            embeddings.values, &fused_hidden).is_ok());
        float step_error = 0.0F;
        CHECK(close(fused_hidden, three_stage_hidden, &step_error));
        fused_hidden_error = std::max(fused_hidden_error, step_error);
    }
    three_stage_context->reset();
    fused_context->reset();
    for (std::size_t step = 0; step < 2; ++step) {
        std::vector<float> three_stage_logits;
        std::vector<float> fused_logits;
        CHECK(three_stage_context->forward_token(kTokenId, &three_stage_logits).is_ok());
        CHECK(fused_context->forward_token(kTokenId, &fused_logits).is_ok());
        float step_error = 0.0F;
        CHECK(close(fused_logits, three_stage_logits, &step_error));
        fused_logits_error = std::max(fused_logits_error, step_error);
        CHECK(std::max_element(three_stage_logits.begin(), three_stage_logits.end()) -
                  three_stage_logits.begin() ==
              std::max_element(fused_logits.begin(), fused_logits.end()) -
                  fused_logits.begin());
    }

    auto sequential_prefill_context = cuda_model->create_context(2, &error);
    auto tensor_prefill_context = cuda_model->create_context(2, &error);
    CHECK(sequential_prefill_context != nullptr && tensor_prefill_context != nullptr);
    tensor_prefill_context->set_attention_path(
        cpp_llm_engine::cuda::CudaAttentionPath::FusedOnline);
    std::vector<float> sequence_input = embeddings.values;
    sequence_input.insert(
        sequence_input.end(), embeddings.values.begin(), embeddings.values.end());
    std::vector<float> sequential_prefill_hidden;
    for (std::size_t step = 0; step < 2; ++step) {
        std::vector<float> row;
        CHECK(sequential_prefill_context->forward_hidden(
            embeddings.values, &row).is_ok());
        sequential_prefill_hidden.insert(
            sequential_prefill_hidden.end(), row.begin(), row.end());
    }
    std::vector<float> tensor_prefill_hidden;
    CHECK(tensor_prefill_context->forward_hidden_sequence(
        sequence_input, 2, &tensor_prefill_hidden).is_ok());
    float tensor_prefill_error = 0.0F;
    CHECK(close(
        tensor_prefill_hidden, sequential_prefill_hidden,
        &tensor_prefill_error));
    CHECK(tensor_prefill_context->position() == 2);

    auto sequential_token_prefill = cuda_model->create_context(2, &error);
    auto tensor_token_prefill = cuda_model->create_context(2, &error);
    CHECK(sequential_token_prefill != nullptr && tensor_token_prefill != nullptr);
    tensor_token_prefill->set_attention_path(
        cpp_llm_engine::cuda::CudaAttentionPath::FusedOnline);
    std::vector<float> sequential_token_logits;
    CHECK(sequential_token_prefill->forward_token(
        kTokenId, &sequential_token_logits).is_ok());
    CHECK(sequential_token_prefill->forward_token(
        kTokenId, &sequential_token_logits).is_ok());
    std::vector<float> tensor_token_logits;
    CHECK(tensor_token_prefill->prefill_tokens(
        {kTokenId, kTokenId}, &tensor_token_logits).is_ok());
    float token_prefill_error = 0.0F;
    CHECK(close(
        tensor_token_logits, sequential_token_logits,
        &token_prefill_error));
    CHECK(std::max_element(tensor_token_logits.begin(), tensor_token_logits.end()) -
              tensor_token_logits.begin() ==
          std::max_element(sequential_token_logits.begin(), sequential_token_logits.end()) -
              sequential_token_logits.begin());
    CHECK(tensor_token_prefill->position() == 2);
    CHECK(tensor_token_prefill->prefill_workspace_creation_count() == 1);
    tensor_token_prefill->reset();
    CHECK(tensor_token_prefill->prefill_tokens(
        {kTokenId, kTokenId}, &tensor_token_logits).is_ok());
    CHECK(tensor_token_prefill->prefill_workspace_creation_count() == 1);

    std::cout << "cuda_multilayer_match: true\n"
              << "layer_count: " << layer_count << '\n'
              << "hidden_max_error: " << hidden_error << '\n'
              << "key_max_error: " << maximum_key_error << '\n'
              << "value_max_error: " << maximum_value_error << '\n'
              << "logits_max_error: " << logits_error << '\n'
              << "fused_hidden_max_error: " << fused_hidden_error << '\n'
              << "fused_logits_max_error: " << fused_logits_error << '\n'
              << "tensor_prefill_max_error: " << tensor_prefill_error << '\n'
              << "token_prefill_max_error: " << token_prefill_error << '\n'
              << "prefill_workspace_creations: "
              << tensor_token_prefill->prefill_workspace_creation_count() << '\n'
              << "greedy_token: "
              << std::distance(cuda_logits.begin(), cuda_greedy) << '\n';
    return 0;
}

#undef CHECK
