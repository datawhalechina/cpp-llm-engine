// Engine CUDA：验证真实 Llama/Qwen3 layer-0 的单 token 前向。
// 职责：用 Chapter05 CPU fused 路径生成独立参考，再核对 CUDA hidden 与 K/V。
// 通过条件：真实权重上传、首层执行、缓存增长和 CPU/CUDA 数值误差均符合契约。
#include "cpp_llm_engine/model/real_llama_cuda_model.h"

#include "chapter05_quantization/q4k_dequantize.h"
#include "chapter05_quantization/q4k_linear.h"
#include "chapter05_quantization/q6k_linear.h"
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::cerr << "check failed line " << __LINE__ << '\n'; return 1; } } while (false)

namespace {

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index, const std::string& name) {
    for (const auto& tensor : index.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool load_named(
    const std::string& path,
    const chapter01::GgufModelIndex& index,
    const std::string& name,
    chapter01::Tensor* output,
    std::string* error) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) {
        *error = "missing tensor: " + name;
        return false;
    }
    return chapter01::GgufTensorLoader::load_tensor(path, index, *info, output, error);
}

bool linear(
    const chapter01::Tensor& weight,
    const std::vector<float>& input,
    std::vector<float>* output,
    std::string* error) {
    if (weight.shape().size() != 2) {
        *error = "Linear weight is not rank 2: " + weight.name();
        return false;
    }
    const auto rows = static_cast<std::size_t>(weight.shape()[1]);
    const auto cols = static_cast<std::size_t>(weight.shape()[0]);
    const auto* payload = reinterpret_cast<const std::uint8_t*>(weight.raw_data());
    if (weight.dtype() == chapter01::DType::Q4_K) {
        return chapter05_quantization::Q4KLinearFused::forward(
            payload, weight.bytes(), rows, cols, input, output, error);
    }
    if (weight.dtype() == chapter01::DType::Q6_K) {
        return chapter05_quantization::Q6KLinearFused::forward(
            payload, weight.bytes(), rows, cols, input, output, error);
    }
    *error = "Linear weight is not Q4_K/Q6_K: " + weight.name();
    return false;
}

std::vector<float> rms_norm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    float epsilon) {
    float square_sum = 0.0F;
    for (float value : input) square_sum += value * value;
    const float inverse = 1.0F /
        std::sqrt(square_sum / static_cast<float>(input.size()) + epsilon);
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * inverse * weight[i];
    }
    return output;
}

bool rms_norm_heads(
    std::vector<float>* values,
    std::size_t head_count,
    std::size_t head_dimension,
    const std::vector<float>& weight,
    float epsilon) {
    if (values == nullptr || values->size() != head_count * head_dimension ||
        weight.size() != head_dimension) {
        return false;
    }
    for (std::size_t head = 0; head < head_count; ++head) {
        float square_sum = 0.0F;
        const std::size_t offset = head * head_dimension;
        for (std::size_t index = 0; index < head_dimension; ++index) {
            const float value = (*values)[offset + index];
            square_sum += value * value;
        }
        const float inverse = 1.0F / std::sqrt(
            square_sum / static_cast<float>(head_dimension) + epsilon);
        for (std::size_t index = 0; index < head_dimension; ++index) {
            (*values)[offset + index] *= inverse * weight[index];
        }
    }
    return true;
}

bool close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float* max_error) {
    if (actual.size() != expected.size()) return false;
    float maximum = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        maximum = std::max(maximum, error);
        if (!std::isfinite(actual[i]) ||
            error > 3.0e-2F + 5.0e-3F * std::fabs(expected[i])) {
            if (max_error != nullptr) *max_error = maximum;
            return false;
        }
    }
    if (max_error != nullptr) *max_error = maximum;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_tinyllama_cuda_first_layer_test <dense-model.gguf>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);

    const std::string model_path = argv[1];
    std::string error;
    chapter01::GgufModelIndex index;
    CHECK(chapter01::GgufReader::read_index(model_path, &index, &error));
    chapter01::ModelMetadata metadata;
    CHECK(chapter01::ModelMetadataAdapter::from_gguf(index.metadata, &metadata, &error));
    chapter01::LlamaTensorLayout layout;
    CHECK(chapter01::ModelSchema::validate_gguf(metadata, index, &layout, &error));

    chapter01::Tensor embedding;
    CHECK(load_named(model_path, index, layout.embedding, &embedding, &error));
    chapter01::RealLlamaLayerWeights host_weights;
    CHECK(chapter01::RealLlamaLayerLoader::load(
        model_path, index, layout, 0, &host_weights, &error));

    constexpr std::size_t kTokenId = 15043;
    constexpr std::size_t kQ4BlockElements = 256;
    constexpr std::size_t kQ4BlockBytes = 144;
    CHECK(embedding.dtype() == chapter01::DType::Q4_K);
    const std::size_t embedding_row_bytes =
        metadata.hidden_size / kQ4BlockElements * kQ4BlockBytes;
    const std::size_t embedding_offset = kTokenId * embedding_row_bytes;
    CHECK(embedding_offset + embedding_row_bytes <= embedding.bytes());
    std::vector<float> initial_hidden;
    CHECK(chapter05_quantization::dequantize_q4_k(
        reinterpret_cast<const std::uint8_t*>(embedding.raw_data()) + embedding_offset,
        embedding_row_bytes, metadata.hidden_size, &initial_hidden, &error));

    const auto attention_norm = host_weights.attn_norm.to_vector<float>();
    const auto ffn_norm = host_weights.ffn_norm.to_vector<float>();
    const auto normalized = rms_norm(
        initial_hidden, attention_norm, metadata.rms_norm_epsilon);
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    CHECK(linear(host_weights.attn_q, normalized, &query, &error));
    CHECK(linear(host_weights.attn_k, normalized, &key, &error));
    CHECK(linear(host_weights.attn_v, normalized, &value, &error));

    const std::size_t query_heads = metadata.attention_head_count;
    const std::size_t kv_heads = metadata.attention_head_count_kv;
    CHECK(query_heads > 0 && kv_heads > 0 && query_heads % kv_heads == 0);
    const std::size_t head_dimension = metadata.attention_key_length == 0
        ? metadata.hidden_size / query_heads : metadata.attention_key_length;
    CHECK(metadata.attention_value_length == 0 ||
          metadata.attention_value_length == head_dimension);
    if (host_weights.attn_q_norm.bytes() > 0) {
        CHECK(rms_norm_heads(
            &query, query_heads, head_dimension,
            host_weights.attn_q_norm.to_vector<float>(),
            metadata.rms_norm_epsilon));
        CHECK(rms_norm_heads(
            &key, kv_heads, head_dimension,
            host_weights.attn_k_norm.to_vector<float>(),
            metadata.rms_norm_epsilon));
    }
    std::vector<float> attention(query_heads * head_dimension, 0.0F);
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::size_t kv_head = query_head / (query_heads / kv_heads);
        std::copy_n(
            value.begin() + kv_head * head_dimension,
            head_dimension,
            attention.begin() + query_head * head_dimension);
    }
    std::vector<float> attention_projection;
    CHECK(linear(
        host_weights.attn_output, attention, &attention_projection, &error));
    std::vector<float> expected = initial_hidden;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] += attention_projection[i];
    }
    const auto ffn_input = rms_norm(expected, ffn_norm, metadata.rms_norm_epsilon);
    std::vector<float> gate;
    std::vector<float> up;
    CHECK(linear(host_weights.ffn_gate, ffn_input, &gate, &error));
    CHECK(linear(host_weights.ffn_up, ffn_input, &up, &error));
    for (std::size_t i = 0; i < gate.size(); ++i) {
        gate[i] = gate[i] / (1.0F + std::exp(-gate[i])) * up[i];
    }
    std::vector<float> ffn_projection;
    CHECK(linear(host_weights.ffn_down, gate, &ffn_projection, &error));
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] += ffn_projection[i];

    auto model = cpp_llm_engine::model::RealLlamaCudaModel::load(
        model_path, 1, &error);
    CHECK(model != nullptr);
    CHECK(model->layer_count() == 1);
    CHECK(model->model_layer_count() == metadata.layer_count);
    CHECK(model->hidden_size() == metadata.hidden_size);
    auto context = model->create_context(1, &error);
    CHECK(context != nullptr);
    std::vector<float> actual;
    CHECK(context->forward_hidden(initial_hidden, &actual).is_ok());
    CHECK(context->position() == 1);
    std::vector<float> actual_key;
    std::vector<float> actual_value;
    CHECK(context->copy_layer_kv(0, &actual_key, &actual_value).is_ok());
    float hidden_error = 0.0F;
    float key_error = 0.0F;
    float value_error = 0.0F;
    CHECK(close(actual, expected, &hidden_error));
    CHECK(close(actual_key, key, &key_error));
    CHECK(close(actual_value, value, &value_error));
    CHECK(!context->forward_hidden(initial_hidden, &actual).is_ok());
    CHECK(context->position() == 1);
    context->reset();
    CHECK(context->position() == 0);
    std::vector<float> repeated;
    CHECK(context->forward_hidden(initial_hidden, &repeated).is_ok());
    CHECK(close(repeated, expected, nullptr));
    auto isolated = model->create_context(1, &error);
    CHECK(isolated != nullptr && isolated->position() == 0);
    std::vector<float> isolated_output;
    CHECK(isolated->forward_hidden(initial_hidden, &isolated_output).is_ok());
    CHECK(close(isolated_output, expected, nullptr));
    std::cout << "cuda_first_layer_match: true\n"
              << "hidden_max_error: " << hidden_error << '\n'
              << "key_max_error: " << key_error << '\n'
              << "value_max_error: " << value_error << '\n'
              << "request_context_isolation: true\n";
    return 0;
}

#undef CHECK
