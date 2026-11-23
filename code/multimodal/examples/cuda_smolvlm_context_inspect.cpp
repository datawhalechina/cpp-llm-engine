// 多模态 CUDA：真实视觉 embedding 接入 SmolVLM CUDA 语言 Context。
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"
#include "cpp_llm_engine/model/real_llama_cuda_model.h"
#include "multimodal/cuda_vision_pipeline.h"
#include "multimodal/image.h"
#include "multimodal/image_preprocessor.h"
#include "multimodal/mmproj_schema.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: multimodal_cuda_smolvlm_context_inspect "
                     "<mmproj.gguf> <model.gguf> <image>\n";
        return 2;
    }
    const std::string mmproj_path = argv[1];
    const std::string model_path = argv[2];

    chapter01::GgufModelIndex index;
    multimodal::VisionModelConfig config;
    multimodal::VisionTensorLayout layout;
    std::string error;
    if (!chapter01::GgufReader::read_index(mmproj_path, &index, &error) ||
        !multimodal::MmprojSchema::config_from_gguf(
            index.metadata, &config, &error) ||
        !multimodal::MmprojSchema::validate_gguf(
            config, index, &layout, &error)) {
        std::cerr << "mmproj schema failed: " << error << '\n';
        return 1;
    }
    if (config.image_size > static_cast<std::size_t>(
            std::numeric_limits<int>::max()) ||
        config.patch_size > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        std::cerr << "vision shape exceeds preprocessor range\n";
        return 1;
    }

    multimodal::RgbImage image;
    if (!multimodal::load_rgb_image(argv[3], &image, &error)) {
        std::cerr << "image load failed: " << error << '\n';
        return 1;
    }
    const multimodal::ImagePreprocessorConfig preprocess{
        static_cast<int>(config.image_size),
        static_cast<int>(config.image_size),
        static_cast<int>(config.patch_size),
        static_cast<int>(config.patch_size),
        config.image_mean, config.image_std};
    multimodal::PreprocessedImage prepared;
    if (!multimodal::preprocess_image(
            image, preprocess, &prepared, &error)) {
        std::cerr << "image preprocess failed: " << error << '\n';
        return 1;
    }

    const std::size_t source_grid = config.image_size / config.patch_size;
    const std::size_t selected_grid = config.merge_size;
    multimodal::PatchTensor selected;
    selected.patch_count = static_cast<int>(selected_grid * selected_grid);
    selected.patch_values = prepared.patches.patch_values;
    std::vector<std::size_t> positions;
    for (std::size_t y = 0; y < selected_grid; ++y) {
        for (std::size_t x = 0; x < selected_grid; ++x) {
            const std::size_t position = y * source_grid + x;
            positions.push_back(position);
            const auto begin = prepared.patches.values.begin() +
                position * static_cast<std::size_t>(selected.patch_values);
            selected.values.insert(
                selected.values.end(), begin, begin + selected.patch_values);
        }
    }

    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;
    CudaExecutionContext vision_context;
    multimodal::CudaSmolvlmVisionPipeline pipeline;
    CudaTensor visual_output;
    auto status = CudaExecutionContext::create(&vision_context);
    if (status.is_ok()) {
        status = multimodal::CudaSmolvlmVisionPipeline::upload_gguf(
            &vision_context, mmproj_path, index, config, layout, &pipeline);
    }
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {1, config.projection_dim}, DType::Float32, &visual_output);
    if (status.is_ok()) status = pipeline.forward(
        &vision_context, selected, positions, selected_grid, selected_grid,
        &visual_output);
    std::vector<float> visual_embedding(config.projection_dim);
    if (status.is_ok()) status = vision_context.download(
        visual_output, visual_embedding.data(),
        visual_embedding.size() * sizeof(float));
    if (status.is_ok()) status = vision_context.synchronize();
    if (!status.is_ok()) {
        std::cerr << "CUDA vision pipeline failed: " << status.message() << '\n';
        return 1;
    }

    auto cpu_model = chapter01::RealLlamaCpuModel::load(model_path, 0, &error);
    auto cuda_model = cpp_llm_engine::model::RealLlamaCudaModel::load(
        model_path, 0, &error);
    if (cpu_model == nullptr || cuda_model == nullptr ||
        cuda_model->hidden_size() != config.projection_dim) {
        std::cerr << "SmolVLM decoder load failed: " << error << '\n';
        return 1;
    }
    auto cpu_context = cpu_model->create_context();
    auto cuda_context = cuda_model->create_context(4, &error);
    if (cpu_context == nullptr || cuda_context == nullptr) {
        std::cerr << "SmolVLM context creation failed: " << error << '\n';
        return 1;
    }
    const cpp_llm_engine::core::DecoderEmbeddingSequence embeddings{
        1, config.projection_dim, visual_embedding};
    std::vector<float> cpu_logits;
    std::vector<float> cuda_logits;
    const auto cpu_status = cpu_context->prefill_embeddings(
        embeddings, &cpu_logits);
    const auto cuda_status = cuda_context->prefill_embeddings(
        embeddings, &cuda_logits);
    if (!cpu_status.is_ok() || !cuda_status.is_ok()) {
        std::cerr << "SmolVLM embedding prefill failed: "
                  << (cpu_status.is_ok() ? cuda_status.message()
                                         : cpu_status.message())
                  << '\n';
        return 1;
    }

    float maximum_error = 0.0F;
    double dot = 0.0;
    double cpu_square = 0.0;
    double cuda_square = 0.0;
    for (std::size_t value = 0; value < cpu_logits.size(); ++value) {
        maximum_error = std::max(
            maximum_error,
            std::fabs(cpu_logits[value] - cuda_logits[value]));
        dot += static_cast<double>(cpu_logits[value]) * cuda_logits[value];
        cpu_square += static_cast<double>(cpu_logits[value]) * cpu_logits[value];
        cuda_square += static_cast<double>(cuda_logits[value]) * cuda_logits[value];
    }
    const double cosine = dot / std::sqrt(cpu_square * cuda_square);
    const auto cpu_top = std::max_element(cpu_logits.begin(), cpu_logits.end());
    const auto cuda_top = std::max_element(cuda_logits.begin(), cuda_logits.end());
    const auto cpu_token = std::distance(cpu_logits.begin(), cpu_top);
    const auto cuda_token = std::distance(cuda_logits.begin(), cuda_top);
    const bool matched = cpu_logits.size() == cuda_logits.size() &&
        std::isfinite(maximum_error) && std::isfinite(cosine) &&
        cosine >= 0.999 && cpu_token == cuda_token &&
        cpu_context->token_count() == 1 && cuda_context->position() == 1;
    std::cout << std::setprecision(10)
              << "decoder_layers: " << cuda_model->layer_count() << '\n'
              << "visual_tokens: 1\n"
              << "hidden_size: " << config.projection_dim << '\n'
              << "logits_maximum_error: " << maximum_error << '\n'
              << "logits_cosine_similarity: " << cosine << '\n'
              << "cpu_greedy_token: " << cpu_token << '\n'
              << "cuda_greedy_token: " << cuda_token << '\n'
              << "context_positions: " << cpu_context->token_count() << ','
              << cuda_context->position() << '\n'
              << "cpu_cuda_match: " << (matched ? "true" : "false") << '\n';
    return matched ? 0 : 1;
}
