// 多模态 CUDA：真实图片视觉流水线 CPU/CUDA 对照。
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "multimodal/cuda_vision_pipeline.h"
#include "multimodal/image.h"
#include "multimodal/image_preprocessor.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_projector.h"
#include "multimodal/vision_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: multimodal_cuda_vision_pipeline_inspect "
                     "<mmproj.gguf> <image> [--full-tile]\n";
        return 2;
    }
    const bool full_tile = argc == 4 && std::string(argv[3]) == "--full-tile";
    if (argc == 4 && !full_tile) {
        std::cerr << "unknown vision pipeline option: " << argv[3] << '\n';
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
    if (config.image_size > static_cast<std::size_t>(
            std::numeric_limits<int>::max()) ||
        config.patch_size > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        std::cerr << "vision shape exceeds preprocessor range\n";
        return 1;
    }

    multimodal::RgbImage image;
    if (!multimodal::load_rgb_image(argv[2], &image, &error)) {
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
    const std::size_t selected_grid =
        full_tile ? source_grid : config.merge_size;
    const auto cpu_backend = full_tile
        ? multimodal::VisionComputeBackend::Cblas
        : multimodal::VisionComputeBackend::ScalarReference;
    if (!multimodal::vision_compute_backend_available(cpu_backend)) {
        std::cerr << "full-tile CPU reference requires the CBLAS backend\n";
        return 1;
    }
    std::vector<std::size_t> positions;
    multimodal::PatchTensor selected;
    selected.patch_count = static_cast<int>(selected_grid * selected_grid);
    selected.patch_values = prepared.patches.patch_values;
    selected.values.reserve(
        static_cast<std::size_t>(selected.patch_count) *
        static_cast<std::size_t>(selected.patch_values));
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

    multimodal::PatchEmbeddingWeights patch_weights;
    multimodal::LayerNormWeights post_norm;
    multimodal::LinearWeights projector_weights;
    if (!multimodal::load_patch_embedding_weights(
            argv[1], index, config, layout, &patch_weights, &error) ||
        !multimodal::load_post_layer_norm_weights(
            argv[1], index, config, layout, &post_norm, &error) ||
        !multimodal::load_idefics3_projector_weights(
            argv[1], index, config, layout, &projector_weights, &error)) {
        std::cerr << "CPU weight load failed: " << error << '\n';
        return 1;
    }
    multimodal::FloatMatrix embedding;
    if (!multimodal::embed_patches_at_positions_with_backend(
            selected, positions, patch_weights, cpu_backend,
            &embedding, &error)) {
        std::cerr << "CPU embedding failed: " << error << '\n';
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
    multimodal::FloatMatrix encoded;
    std::vector<multimodal::SiglipLayerSummary> summaries;
    multimodal::SiglipEncoderStats stats;
    if (!multimodal::run_siglip_encoder_with_backend(
            embedding, config.block_count, config.attention_head_count,
            config.layer_norm_epsilon, provider, post_norm, cpu_backend, &encoded,
            &summaries, &stats, &error)) {
        std::cerr << "CPU encoder failed: " << error << '\n';
        return 1;
    }
    multimodal::Idefics3ProjectorTrace expected;
    if (!multimodal::run_idefics3_projector_with_backend(
            encoded, selected_grid, selected_grid, config.merge_size,
            projector_weights, cpu_backend, &expected, &error)) {
        std::cerr << "CPU projector failed: " << error << '\n';
        return 1;
    }

    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;
    CudaExecutionContext context;
    multimodal::CudaSmolvlmVisionPipeline pipeline;
    CudaTensor output;
    auto status = CudaExecutionContext::create(&context);
    if (status.is_ok()) {
        status = multimodal::CudaSmolvlmVisionPipeline::upload_gguf(
            &context, path, index, config, layout, &pipeline);
    }
    if (status.is_ok()) status = CudaTensor::create_contiguous(
        {expected.output.rows, expected.output.cols}, DType::Float32, &output);
    if (status.is_ok()) status = pipeline.forward(
        &context, selected, positions, selected_grid, selected_grid, &output);
    std::vector<float> actual(expected.output.values.size());
    if (status.is_ok()) status = context.download(
        output, actual.data(), actual.size() * sizeof(float));
    if (status.is_ok()) status = context.synchronize();
    if (!status.is_ok()) {
        std::cerr << "CUDA vision pipeline failed: " << status.message() << '\n';
        return 1;
    }

    float maximum_error = 0.0F;
    double dot = 0.0;
    double expected_square = 0.0;
    double actual_square = 0.0;
    for (std::size_t value = 0; value < actual.size(); ++value) {
        maximum_error = std::max(
            maximum_error,
            std::fabs(actual[value] - expected.output.values[value]));
        dot += static_cast<double>(actual[value]) * expected.output.values[value];
        expected_square += static_cast<double>(expected.output.values[value]) *
            expected.output.values[value];
        actual_square += static_cast<double>(actual[value]) * actual[value];
    }
    const double cosine = dot / std::sqrt(expected_square * actual_square);
    const bool matched = std::isfinite(maximum_error) &&
        std::isfinite(cosine) && maximum_error <= 5.0e-2F &&
        cosine >= 0.9999;
    std::cout << std::setprecision(10)
              << "source_image: " << image.width << 'x' << image.height << '\n'
              << "vision_mode: "
              << (full_tile ? "full-tile" : "top-left-window") << '\n'
              << "cpu_backend: "
              << (full_tile ? "cblas" : "scalar") << '\n'
              << "selected_grid: " << selected_grid << 'x' << selected_grid << '\n'
              << "positions: ";
    if (full_tile) {
        std::cout << positions.front() << ".." << positions.back();
    } else {
        for (std::size_t index_value = 0; index_value < positions.size();
             ++index_value) {
            if (index_value != 0) std::cout << ',';
            std::cout << positions[index_value];
        }
    }
    std::cout << '\n'
              << "output_shape: " << expected.output.rows << 'x'
              << expected.output.cols << '\n'
              << "workspace_rebuilds: "
              << pipeline.workspace_rebuild_count() << '\n'
              << "maximum_error: " << maximum_error << '\n'
              << "cosine_similarity: " << cosine << '\n'
              << "cpu_cuda_match: " << (matched ? "true" : "false") << '\n';
    return matched ? 0 : 1;
}
