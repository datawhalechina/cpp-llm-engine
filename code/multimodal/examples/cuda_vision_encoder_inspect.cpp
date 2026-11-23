// 多模态 CUDA：真实 mmproj 12 层 SigLIP Encoder CPU/CUDA 对照。
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "multimodal/cuda_vision_encoder.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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
        std::cerr << "usage: multimodal_cuda_vision_encoder_inspect "
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
    multimodal::FloatMatrix input;
    input.rows = token_count;
    input.cols = config.hidden_size;
    input.values.resize(token_count * config.hidden_size);
    for (std::size_t value = 0; value < input.values.size(); ++value) {
        input.values[value] =
            std::sin(static_cast<float>(value + 1) * 0.013F) * 0.25F;
    }

    multimodal::LayerNormWeights post_norm;
    if (!multimodal::load_post_layer_norm_weights(
            argv[1], index, config, layout, &post_norm, &error)) {
        std::cerr << "post norm load failed: " << error << '\n';
        return 1;
    }
    const std::string path = argv[1];
    const multimodal::SiglipBlockWeightProvider provider =
        [&path, &index, &config, &layout](
            std::size_t layer, multimodal::SiglipBlockWeights* weights,
            std::string* message) {
            return multimodal::load_siglip_block_weights(
                path, index, config, layout, layer, weights, message);
        };
    multimodal::FloatMatrix expected;
    std::vector<multimodal::SiglipLayerSummary> summaries;
    multimodal::SiglipEncoderStats stats;
    if (!multimodal::run_siglip_encoder(
            input, config.block_count, config.attention_head_count,
            config.layer_norm_epsilon, provider, post_norm, &expected,
            &summaries, &stats, &error)) {
        std::cerr << "CPU encoder failed: " << error << '\n';
        return 1;
    }

    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;
    CudaExecutionContext context;
    multimodal::CudaSiglipEncoder encoder;
    CudaTensor device_input;
    CudaTensor device_output;
    auto status = CudaExecutionContext::create(&context);
    if (status.is_ok()) status = multimodal::CudaSiglipEncoder::upload_gguf(
        &context, path, index, config, layout, &encoder);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {token_count, config.hidden_size}, DType::Float32, &device_input);
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {token_count, config.hidden_size}, DType::Float32, &device_output);
    if (status.is_ok()) status = context.upload(
        &device_input, input.values.data(), input.values.size() * sizeof(float));
    if (status.is_ok()) status = encoder.forward(
        &context, device_input, &device_output);
    std::vector<float> actual(expected.values.size());
    if (status.is_ok()) status = context.download(
        device_output, actual.data(), actual.size() * sizeof(float));
    if (status.is_ok()) status = context.synchronize();
    if (!status.is_ok()) {
        std::cerr << "CUDA encoder failed: " << status.message() << '\n';
        return 1;
    }

    float maximum_error = 0.0F;
    double dot = 0.0;
    double expected_square = 0.0;
    double actual_square = 0.0;
    for (std::size_t value = 0; value < actual.size(); ++value) {
        maximum_error = std::max(
            maximum_error, std::fabs(actual[value] - expected.values[value]));
        dot += static_cast<double>(actual[value]) * expected.values[value];
        expected_square += static_cast<double>(expected.values[value]) *
            expected.values[value];
        actual_square += static_cast<double>(actual[value]) * actual[value];
    }
    const double cosine = dot / std::sqrt(expected_square * actual_square);
    const bool matched = std::isfinite(maximum_error) &&
        std::isfinite(cosine) && maximum_error <= 5.0e-2F &&
        cosine >= 0.9999;
    std::cout << std::setprecision(10)
              << "tokens: " << token_count << '\n'
              << "hidden: " << config.hidden_size << '\n'
              << "blocks: " << encoder.block_count() << '\n'
              << "weight_format: Q8_0\n"
              << "workspace_rebuilds: " << encoder.workspace_rebuild_count() << '\n'
              << "maximum_error: " << maximum_error << '\n'
              << "cosine_similarity: " << cosine << '\n'
              << "cpu_cuda_match: " << (matched ? "true" : "false") << '\n';
    return matched ? 0 : 1;
}
