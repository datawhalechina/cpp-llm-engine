// Engine CUDA: compare a real DeepSeek2 MLA and MoE layer with CPU.
#include "cpp_llm_engine/model/cuda_deepseek2_moe_layer.h"
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
        std::cerr << "usage: real_deepseek2_cuda_moe_layer_test <model.gguf>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);

    chapter01::GgufModelIndex index;
    chapter01::ModelMetadata metadata;
    chapter01::DeepSeek2DecoderTensorLayout layout;
    chapter01::RealDeepSeek2MoeLayerWeights host_weights;
    std::string error;
    CHECK(chapter01::GgufReader::read_index(argv[1], &index, &error));
    CHECK(chapter01::ModelMetadataAdapter::from_gguf(
        index.metadata, &metadata, &error));
    CHECK(chapter01::ModelSchema::validate_deepseek2_gguf(
        metadata, index, &layout, &error));
    CHECK(chapter01::RealDeepSeek2MoeLayerLoader::load(
        argv[1], index, layout, 1, &host_weights, &error));

    cpp_llm_engine::cuda::CudaExecutionContext context;
    cpp_llm_engine::model::CudaDeepSeek2MoeWeights device_weights;
    CHECK(cpp_llm_engine::cuda::CudaExecutionContext::create(&context).is_ok());
    CHECK(cpp_llm_engine::model::CudaDeepSeek2MoeWeights::upload(
        &context, host_weights, &device_weights).is_ok());

    const std::size_t nope =
        metadata.attention_key_length_mla - metadata.rope_dimension_count;
    float attention_scale = 1.0F / std::sqrt(static_cast<float>(
        metadata.attention_key_length_mla));
    if (metadata.rope_scaling_type == "yarn" &&
        metadata.rope_scaling_factor > 1.0F &&
        metadata.rope_yarn_log_multiplier > 0.0F) {
        const float multiplier = metadata.rope_yarn_log_multiplier / 0.1F;
        const float mscale = 1.0F + 0.1F * multiplier *
            std::log(metadata.rope_scaling_factor);
        attention_scale *= mscale * mscale;
    }
    const cpp_llm_engine::model::CudaDeepSeek2DenseLayerConfig attention{
        metadata.hidden_size, metadata.attention_head_count, nope,
        metadata.rope_dimension_count, metadata.attention_value_length_mla,
        metadata.attention_kv_lora_rank, metadata.feed_forward_length,
        metadata.rms_norm_epsilon, metadata.rope_freq_base,
        {metadata.rope_scaling_factor,
         metadata.rope_scaling_original_context, 32.0F, 1.0F},
        attention_scale};
    const cpp_llm_engine::model::CudaDeepSeek2MoeConfig moe{
        metadata.hidden_size, device_weights.expert_hidden_size(),
        device_weights.shared_expert_hidden_size(), metadata.expert_count,
        metadata.expert_used_count, metadata.expert_weights_normalize,
        metadata.expert_weights_scale};
    cpp_llm_engine::model::CudaDeepSeek2MoeLayer layer;
    CHECK(cpp_llm_engine::model::CudaDeepSeek2MoeLayer::create(
        {attention, moe}, 2, &layer).is_ok());

    cpp_llm_engine::cuda::CudaTensor hidden;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {metadata.hidden_size}, cpp_llm_engine::core::DType::Float32,
        &hidden).is_ok());
    cpp_llm_engine::model::RealDeepSeek2LayerContext cpu_context;
    float maximum_error = 0.0F;
    double error_sum = 0.0;
    std::size_t compared = 0;
    for (std::size_t token = 0; token < 2; ++token) {
        std::vector<float> input(metadata.hidden_size);
        for (std::size_t value = 0; value < input.size(); ++value) {
            input[value] = static_cast<float>(
                static_cast<int>((value + token * 13) % 31) - 15) * 0.002F;
        }
        cpp_llm_engine::model::RealDeepSeek2LayerResult cpu_result;
        cpp_llm_engine::model::RealMoeRoute cpu_route;
        CHECK(cpp_llm_engine::model::run_deepseek2_moe_token_layer(
            metadata, host_weights, input, &cpu_context, &cpu_result,
            &cpu_route, &error));
        CHECK(context.upload(
            &hidden, input.data(), input.size() * sizeof(float)).is_ok());
        cpp_llm_engine::model::CudaDeepSeek2MoeRoute cuda_route;
        CHECK(layer.forward(
            &context, device_weights, &hidden, token, &cuda_route).is_ok());
        CHECK(cuda_route.expert_indices == cpu_route.expert_indices);
        std::vector<float> cuda_output(input.size());
        CHECK(context.download(
            hidden, cuda_output.data(), cuda_output.size() * sizeof(float))
            .is_ok());
        CHECK(context.synchronize().is_ok());
        for (std::size_t value = 0; value < cuda_output.size(); ++value) {
            const float difference = std::fabs(
                cuda_output[value] - cpu_result.output[value]);
            maximum_error = std::max(maximum_error, difference);
            error_sum += difference;
            ++compared;
        }
    }
    const double mean_error = error_sum / static_cast<double>(compared);
    std::cout << "deepseek_cuda_moe_layer_max_error=" << maximum_error << '\n'
              << "deepseek_cuda_moe_layer_mean_error=" << mean_error << '\n'
              << "deepseek_cuda_moe_layer_tokens=" << layer.position() << '\n';
    CHECK(layer.position() == 2);
    CHECK(maximum_error <= 0.03F);
    CHECK(mean_error <= 0.003);
    return 0;
}

#undef CHECK
