// 多模态 CUDA：真实 mmproj 第 0 个 SigLIP Block CPU/CUDA 对照。
// 运行：multimodal_cuda_vision_block_inspect <mmproj.gguf> [token-count]。
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "multimodal/cuda_siglip_block.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool parse_token_count(const char* value, std::size_t* output) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 || parsed > 32) return false;
    *output = static_cast<std::size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: multimodal_cuda_vision_block_inspect "
                     "<mmproj.gguf> [token-count]\n";
        return 2;
    }
    std::size_t token_count = 4;
    if (argc == 3 && !parse_token_count(argv[2], &token_count)) {
        std::cerr << "token-count must be an integer in [1, 32]\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    multimodal::VisionModelConfig config;
    multimodal::VisionTensorLayout layout;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error) ||
        !multimodal::MmprojSchema::config_from_gguf(
            index.metadata, &config, &error) ||
        !multimodal::MmprojSchema::validate_gguf(
            config, index, &layout, &error)) {
        std::cerr << "mmproj schema failed: " << error << '\n';
        return 1;
    }

    multimodal::SiglipBlockWeights cpu_weights;
    if (!multimodal::load_siglip_block_weights(
            argv[1], index, config, layout, 0, &cpu_weights, &error)) {
        std::cerr << "CPU weight load failed: " << error << '\n';
        return 1;
    }
    multimodal::FloatMatrix input;
    input.rows = token_count;
    input.cols = config.hidden_size;
    input.values.resize(token_count * config.hidden_size);
    for (std::size_t index_value = 0; index_value < input.values.size();
         ++index_value) {
        input.values[index_value] =
            std::sin(static_cast<float>(index_value + 1) * 0.013F) * 0.25F;
    }
    multimodal::SiglipBlockTrace cpu_trace;
    if (!multimodal::run_siglip_block(
            input, config.attention_head_count, config.layer_norm_epsilon,
            cpu_weights, &cpu_trace, &error)) {
        std::cerr << "CPU SigLIP Block failed: " << error << '\n';
        return 1;
    }

    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;
    CudaExecutionContext context;
    multimodal::CudaSiglipBlock block;
    CudaTensor device_input;
    CudaTensor device_output;
    auto status = CudaExecutionContext::create(&context);
    if (status.is_ok()) status = multimodal::CudaSiglipBlock::upload_gguf(
        &context, argv[1], index, config, layout, 0, &block);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {token_count, config.hidden_size}, DType::Float32, &device_input);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {token_count, config.hidden_size}, DType::Float32, &device_output);
    if (status.is_ok()) status = context.upload(
        &device_input, input.values.data(),
        input.values.size() * sizeof(float));
    if (status.is_ok()) status = block.forward(
        &context, device_input, &device_output);
    std::vector<float> actual(input.values.size());
    if (status.is_ok()) status = context.download(
        device_output, actual.data(), actual.size() * sizeof(float));
    if (status.is_ok()) status = context.synchronize();
    if (!status.is_ok()) {
        std::cerr << "CUDA SigLIP Block failed: " << status.message() << '\n';
        return 1;
    }

    float maximum_error = 0.0F;
    double dot = 0.0;
    double cpu_square = 0.0;
    double cuda_square = 0.0;
    for (std::size_t value = 0; value < actual.size(); ++value) {
        maximum_error = std::max(
            maximum_error,
            std::fabs(actual[value] - cpu_trace.output.values[value]));
        dot += static_cast<double>(actual[value]) *
            cpu_trace.output.values[value];
        cpu_square += static_cast<double>(cpu_trace.output.values[value]) *
            cpu_trace.output.values[value];
        cuda_square += static_cast<double>(actual[value]) * actual[value];
    }
    const double cosine = dot / std::sqrt(cpu_square * cuda_square);
    const bool matched = std::isfinite(maximum_error) &&
        std::isfinite(cosine) && maximum_error <= 2.0e-2F &&
        cosine >= 0.99999;
    std::cout << std::setprecision(10)
              << "tokens: " << token_count << '\n'
              << "hidden: " << config.hidden_size << '\n'
              << "block: 0\n"
              << "weight_format: Q8_0\n"
              << "maximum_error: " << maximum_error << '\n'
              << "cosine_similarity: " << cosine << '\n'
              << "cpu_cuda_match: " << (matched ? "true" : "false") << '\n';
    return matched ? 0 : 1;
}
