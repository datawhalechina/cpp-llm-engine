// Engine CUDA: compare a real DeepSeek2 MoE FFN with the CPU reference.
#include "cpp_llm_engine/model/cuda_deepseek2_moe_feed_forward.h"
#include "cpp_llm_engine/model/real_moe_feed_forward.h"

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
        std::cerr << "usage: real_deepseek2_cuda_moe_feed_forward_test <model.gguf>\n";
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

    std::vector<float> hidden(metadata.hidden_size);
    for (std::size_t value = 0; value < hidden.size(); ++value) {
        hidden[value] = static_cast<float>(
            static_cast<int>((value * 17 + 5) % 41) - 20) * 0.003F;
    }
    std::vector<float> cpu_output;
    cpp_llm_engine::model::RealMoeRoute cpu_route;
    CHECK(cpp_llm_engine::model::run_real_deepseek2_moe_feed_forward(
        metadata, host_weights, hidden, &cpu_output, &cpu_route, &error));

    cpp_llm_engine::cuda::CudaExecutionContext context;
    cpp_llm_engine::model::CudaDeepSeek2MoeWeights device_weights;
    cpp_llm_engine::model::CudaDeepSeek2MoeFeedForward feed_forward;
    cpp_llm_engine::cuda::CudaTensor input;
    cpp_llm_engine::cuda::CudaTensor output;
    CHECK(cpp_llm_engine::cuda::CudaExecutionContext::create(&context).is_ok());
    CHECK(cpp_llm_engine::model::CudaDeepSeek2MoeWeights::upload(
        &context, host_weights, &device_weights).is_ok());
    const cpp_llm_engine::model::CudaDeepSeek2MoeConfig config{
        metadata.hidden_size, device_weights.expert_hidden_size(),
        device_weights.shared_expert_hidden_size(), metadata.expert_count,
        metadata.expert_used_count, metadata.expert_weights_normalize,
        metadata.expert_weights_scale};
    CHECK(cpp_llm_engine::model::CudaDeepSeek2MoeFeedForward::create(
        config, &feed_forward).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {metadata.hidden_size}, cpp_llm_engine::core::DType::Float32,
        &input).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {metadata.hidden_size}, cpp_llm_engine::core::DType::Float32,
        &output).is_ok());
    CHECK(context.upload(
        &input, hidden.data(), hidden.size() * sizeof(float)).is_ok());
    cpp_llm_engine::model::CudaDeepSeek2MoeRoute cuda_route;
    CHECK(feed_forward.forward(
        &context, device_weights, input, &output, &cuda_route).is_ok());
    std::vector<float> cuda_output(hidden.size());
    CHECK(context.download(
        output, cuda_output.data(), cuda_output.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());

    CHECK(cuda_route.expert_indices == cpu_route.expert_indices);
    CHECK(cuda_route.weights.size() == cpu_route.weights.size());
    float route_error = 0.0F;
    for (std::size_t value = 0; value < cuda_route.weights.size(); ++value) {
        route_error = std::max(route_error, std::fabs(
            cuda_route.weights[value] - cpu_route.weights[value]));
    }
    float maximum_error = 0.0F;
    double error_sum = 0.0;
    for (std::size_t value = 0; value < cuda_output.size(); ++value) {
        const float difference = std::fabs(
            cuda_output[value] - cpu_output[value]);
        maximum_error = std::max(maximum_error, difference);
        error_sum += difference;
    }
    const double mean_error = error_sum /
        static_cast<double>(cuda_output.size());
    std::cout << "deepseek_cuda_moe_route_error=" << route_error << '\n'
              << "deepseek_cuda_moe_max_error=" << maximum_error << '\n'
              << "deepseek_cuda_moe_mean_error=" << mean_error << '\n'
              << "deepseek_cuda_moe_selected_experts=";
    for (std::size_t expert : cuda_route.expert_indices) {
        std::cout << expert << ',';
    }
    std::cout << '\n';
    CHECK(route_error <= 1.0e-6F);
    CHECK(maximum_error <= 0.03F);
    CHECK(mean_error <= 0.003);

    std::vector<float> hidden_b(metadata.hidden_size);
    for (std::size_t value = 0; value < hidden_b.size(); ++value) {
        hidden_b[value] = static_cast<float>(
            static_cast<int>((value * 13 + 9) % 37) - 18) * 0.002F;
    }
    cpp_llm_engine::cuda::CudaTensor input_b;
    cpp_llm_engine::cuda::CudaTensor output_b;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {metadata.hidden_size}, cpp_llm_engine::core::DType::Float32,
        &input_b).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {metadata.hidden_size}, cpp_llm_engine::core::DType::Float32,
        &output_b).is_ok());
    CHECK(context.upload(
        &input_b, hidden_b.data(), hidden_b.size() * sizeof(float)).is_ok());
    cpp_llm_engine::model::CudaDeepSeek2MoeRoute sequential_route_b;
    CHECK(feed_forward.forward(
        &context, device_weights, input_b, &output_b,
        &sequential_route_b).is_ok());
    std::vector<float> sequential_b(hidden_b.size());
    CHECK(context.download(
        output_b, sequential_b.data(),
        sequential_b.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());

    cpp_llm_engine::cuda::CudaTensor batch_output_a;
    cpp_llm_engine::cuda::CudaTensor batch_output_b;
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {metadata.hidden_size}, cpp_llm_engine::core::DType::Float32,
        &batch_output_a).is_ok());
    CHECK(cpp_llm_engine::cuda::CudaTensor::create_contiguous(
        {metadata.hidden_size}, cpp_llm_engine::core::DType::Float32,
        &batch_output_b).is_ok());
    cpp_llm_engine::model::CudaDeepSeek2MoeBatchWorkspace batch_workspace;
    CHECK(cpp_llm_engine::model::CudaDeepSeek2MoeBatchWorkspace::create(
        config, 2, &batch_workspace).is_ok());
    std::vector<cpp_llm_engine::model::CudaDeepSeek2MoeRoute> batch_routes;
    CHECK(batch_workspace.forward(
        &context, device_weights, {&input, &input_b},
        {&batch_output_a, &batch_output_b}, &batch_routes).is_ok());
    CHECK(batch_workspace.batch_size() == 2);
    CHECK(batch_workspace.use_count() == 1);
    CHECK(batch_routes.size() == 2);
    CHECK(batch_routes[0].expert_indices == cuda_route.expert_indices);
    CHECK(batch_routes[1].expert_indices ==
          sequential_route_b.expert_indices);
    std::vector<float> batch_a(hidden.size());
    std::vector<float> batch_b(hidden.size());
    CHECK(context.download(
        batch_output_a, batch_a.data(), batch_a.size() * sizeof(float)).is_ok());
    CHECK(context.download(
        batch_output_b, batch_b.data(), batch_b.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    float batch_error = 0.0F;
    for (std::size_t value = 0; value < batch_a.size(); ++value) {
        batch_error = std::max(batch_error, std::fabs(
            batch_a[value] - cuda_output[value]));
        batch_error = std::max(batch_error, std::fabs(
            batch_b[value] - sequential_b[value]));
    }
    std::cout << "deepseek_cuda_moe_batch_max_error=" << batch_error << '\n'
              << "deepseek_cuda_moe_batch_workspace_uses="
              << batch_workspace.use_count() << '\n';
    CHECK(batch_error <= 0.03F);
    return 0;
}

#undef CHECK
