// Engine CUDA：用真实 DeepSeek-V2-Lite 对照两词元 CPU/CUDA 稠密层。
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer.h"
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer_weights.h"
#include "cpp_llm_engine/model/real_deepseek2_dense_layer.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "gguf_reader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
    std::cerr << "check failed line " << __LINE__ << '\n'; return 1; \
} } while (false)

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_deepseek2_cuda_dense_layer_test <model.gguf>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    chapter01::GgufModelIndex index;
    chapter01::ModelMetadata metadata;
    chapter01::DeepSeek2DecoderTensorLayout layout;
    chapter01::RealDeepSeek2DenseLayerWeights host_weights;
    std::string error;
    CHECK(chapter01::GgufReader::read_index(argv[1], &index, &error));
    CHECK(chapter01::ModelMetadataAdapter::from_gguf(
        index.metadata, &metadata, &error));
    CHECK(chapter01::ModelSchema::validate_deepseek2_gguf(
        metadata, index, &layout, &error));
    CHECK(chapter01::RealDeepSeek2DenseLayerLoader::load(
        argv[1], index, layout, 0, &host_weights, &error));

    const std::size_t nope =
        metadata.attention_key_length_mla - metadata.rope_dimension_count;
    float scale = 1.0F / std::sqrt(static_cast<float>(
        metadata.attention_key_length_mla));
    if (metadata.rope_scaling_type == "yarn" &&
        metadata.rope_scaling_factor > 1.0F &&
        metadata.rope_yarn_log_multiplier > 0.0F) {
        const float multiplier = metadata.rope_yarn_log_multiplier / 0.1F;
        const float mscale = 1.0F + 0.1F * multiplier *
            std::log(metadata.rope_scaling_factor);
        scale *= mscale * mscale;
    }
    const cpp_llm_engine::model::CudaDeepSeek2DenseLayerConfig config{
        metadata.hidden_size, metadata.attention_head_count, nope,
        metadata.rope_dimension_count, metadata.attention_value_length_mla,
        metadata.attention_kv_lora_rank, metadata.feed_forward_length,
        metadata.rms_norm_epsilon, metadata.rope_freq_base,
        {metadata.rope_scaling_factor,
         metadata.rope_scaling_original_context, 32.0F, 1.0F}, scale};

    cpp_llm_engine::cuda::CudaExecutionContext cuda_context;
    cpp_llm_engine::model::CudaDeepSeek2DenseLayerWeights cuda_weights;
    cpp_llm_engine::model::CudaDeepSeek2DenseLayer cuda_layer;
    cpp_llm_engine::cuda::CudaTensor cuda_hidden;
    CHECK(cpp_llm_engine::cuda::CudaExecutionContext::create(
        &cuda_context).is_ok());
    CHECK(cpp_llm_engine::model::CudaDeepSeek2DenseLayerWeights::upload(
        &cuda_context, host_weights, &cuda_weights).is_ok());
    CHECK(cpp_llm_engine::model::CudaDeepSeek2DenseLayer::create(
        config, 2, &cuda_layer).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {metadata.hidden_size}, cpp_llm_engine::core::DType::Float32,
        &cuda_hidden).is_ok());

    cpp_llm_engine::model::RealDeepSeek2LayerContext cpu_context;
    float maximum_error = 0.0F;
    double error_sum = 0.0;
    std::size_t compared = 0;
    for (std::size_t token = 0; token < 2; ++token) {
        std::vector<float> input(metadata.hidden_size);
        for (std::size_t index_value = 0; index_value < input.size(); ++index_value) {
            input[index_value] = static_cast<float>(
                static_cast<int>((index_value + token * 13) % 31) - 15) *
                0.002F;
        }
        cpp_llm_engine::model::RealDeepSeek2LayerResult cpu_result;
        CHECK(cpp_llm_engine::model::run_deepseek2_dense_token_layer(
            metadata, host_weights, input, &cpu_context, &cpu_result, &error));
        CHECK(cuda_context.upload(
            &cuda_hidden, input.data(), input.size() * sizeof(float)).is_ok());
        CHECK(cuda_layer.forward(
            &cuda_context, cuda_weights.view(), &cuda_hidden, token).is_ok());
        std::vector<float> cuda_output(input.size());
        CHECK(cuda_context.download(
            cuda_hidden, cuda_output.data(), cuda_output.size() * sizeof(float))
            .is_ok());
        CHECK(cuda_context.synchronize().is_ok());
        for (std::size_t index_value = 0;
             index_value < cuda_output.size(); ++index_value) {
            const float difference = std::fabs(
                cuda_output[index_value] - cpu_result.output[index_value]);
            maximum_error = std::max(maximum_error, difference);
            error_sum += difference;
            ++compared;
        }
    }
    const double mean_error = error_sum / static_cast<double>(compared);
    std::cout << "deepseek_cuda_dense_layer_max_error=" << maximum_error << '\n'
              << "deepseek_cuda_dense_layer_mean_error=" << mean_error << '\n'
              << "deepseek_cuda_dense_layer_tokens=" << cuda_layer.position() << '\n';
    CHECK(cuda_layer.position() == 2);
    CHECK(maximum_error <= 0.02F);
    CHECK(mean_error <= 0.002);
    return 0;
}

#undef CHECK
