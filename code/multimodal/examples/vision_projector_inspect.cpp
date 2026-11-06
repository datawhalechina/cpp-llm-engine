// 多模态 CPU：真实 IDEFICS3 Projector 检查工具
// 职责：编码局部窗口、完整 tile 或动态 tiles，并接入真实 SmolVLM Decoder。
// 契约：完整视觉模式要求 CBLAS；动态模式按 row-major tiles、overview 的顺序展开。
// 运行：multimodal_vision_projector_inspect <mmproj> <image> [model] [layers] [question] [mode]。
#include "multimodal/image.h"
#include "multimodal/image_preprocessor.h"
#include "multimodal/idefics3_preprocessor.h"
#include "multimodal/input_embedding.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/request_metrics.h"
#include "multimodal/smolvlm_prompt.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_projector.h"
#include "multimodal/vision_weights.h"

#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"
#ifdef CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER
#include "real_tokenizer.h"
#endif
#ifdef CPP_LLM_MULTIMODAL_HAS_SERVICE
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/http_adapter.h"
#include "cpp_llm_engine/embedding_request_executor.h"
#endif
#include "chapter05_quantization/q8_0_dequantize.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class InspectMode {
    TopLeftWindow,
    FullTile,
    DynamicTiles,
    DynamicTilesService,
};

#ifdef CPP_LLM_MULTIMODAL_HAS_SERVICE
class RecordingWriter final : public chapter10_serving::ResponseWriter {
public:
    bool write_headers(const chapter10_serving::ResponseHead& value) override {
        head = value;
        return true;
    }

    chapter10_serving::WriteResult write_event(std::string_view event) override {
        body += event;
        return chapter10_serving::WriteResult::ok;
    }

    void finish() override { finished = true; }
    bool client_connected() const override { return true; }

    chapter10_serving::ResponseHead head;
    std::string body;
    bool finished = false;
};

std::string service_request_body(std::size_t request_id, std::size_t positions) {
    std::ostringstream body;
    body << "{\"request_id\":" << request_id << ",\"prompt_tokens\":[";
    for (std::size_t index = 0; index < positions; ++index) {
        if (index != 0) body << ',';
        body << '0';
    }
    body << "]}";
    return body.str();
}
#endif

double checksum(const multimodal::FloatMatrix& matrix) {
    double result = 0.0;
    for (std::size_t i = 0; i < matrix.values.size(); ++i) {
        result += static_cast<double>((i % 97) + 1) * matrix.values[i];
    }
    return result;
}

std::size_t matrix_bytes(const multimodal::FloatMatrix& matrix) {
    return matrix.values.size() * sizeof(float);
}

std::size_t rgb_bytes(const multimodal::RgbImage& image) {
    return image.pixels.size();
}

std::size_t visual_output_bytes(const std::vector<multimodal::FloatMatrix>& outputs) {
    std::size_t result = 0;
    for (const auto& output : outputs) result += matrix_bytes(output);
    return result;
}

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index, const std::string& name) {
    for (const auto& tensor : index.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool load_boundary_text_embeddings(const std::string& model_path,
                                   std::size_t expected_width,
                                   multimodal::FloatMatrix* text,
                                   chapter01::ModelMetadata* metadata,
                                   std::string* error) {
    chapter01::GgufModelIndex index;
    if (!chapter01::GgufReader::read_index(model_path, &index, error) ||
        !chapter01::ModelMetadataAdapter::from_gguf(index.metadata, metadata, error)) {
        return false;
    }
    const auto* info = find_tensor(index, "token_embd.weight");
    if (metadata->architecture != "llama" || metadata->hidden_size != expected_width ||
        info == nullptr || info->type != 8 || info->dimensions.size() != 2 ||
        info->dimensions[0] != expected_width ||
        info->dimensions[1] != metadata->vocab_size ||
        metadata->bos_token_id < 0 || metadata->eos_token_id < 0) {
        if (error != nullptr) *error = "language model is not a compatible Q8_0 Llama embedding source";
        return false;
    }

    chapter01::Tensor tensor;
    if (!chapter01::GgufTensorLoader::load_tensor(model_path, index, *info, &tensor, error)) {
        return false;
    }
    const std::size_t row_bytes =
        expected_width / chapter05_quantization::kQ8_0BlockElements *
        chapter05_quantization::kQ8_0BlockBytes;
    const auto* payload = reinterpret_cast<const std::uint8_t*>(tensor.raw_data());
    const std::size_t bos = static_cast<std::size_t>(metadata->bos_token_id);
    const std::size_t eos = static_cast<std::size_t>(metadata->eos_token_id);
    std::vector<float> bos_values;
    std::vector<float> eos_values;
    if (!chapter05_quantization::dequantize_q8_0(
            payload + bos * row_bytes, row_bytes, expected_width, &bos_values, error) ||
        !chapter05_quantization::dequantize_q8_0(
            payload + eos * row_bytes, row_bytes, expected_width, &eos_values, error)) {
        return false;
    }
    multimodal::FloatMatrix result{3, expected_width, std::vector<float>(3 * expected_width)};
    std::copy(bos_values.begin(), bos_values.end(), result.values.begin());
    std::copy(eos_values.begin(), eos_values.end(),
              result.values.begin() + 2 * expected_width);
    *text = std::move(result);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::cerr << "usage: multimodal_vision_projector_inspect <mmproj.gguf> <image> "
                     "[model.gguf] [decoder-layers] [question] "
                     "[--full-tile|--dynamic-tiles|--dynamic-tiles-service]\n";
        return 2;
    }
    InspectMode mode = InspectMode::TopLeftWindow;
    if (argc == 7) {
        const std::string mode_argument = argv[6];
        if (mode_argument == "--full-tile") {
            mode = InspectMode::FullTile;
        } else if (mode_argument == "--dynamic-tiles") {
            mode = InspectMode::DynamicTiles;
        } else if (mode_argument == "--dynamic-tiles-service") {
            mode = InspectMode::DynamicTilesService;
        } else {
            std::cerr << "unsupported inspect mode\n";
            return 2;
        }
    }
    const bool complete_visual = mode != InspectMode::TopLeftWindow;
    const bool dynamic_tiles =
        mode == InspectMode::DynamicTiles || mode == InspectMode::DynamicTilesService;
    const bool service_mode = mode == InspectMode::DynamicTilesService;
#ifndef CPP_LLM_MULTIMODAL_HAS_SERVICE
    if (service_mode) {
        std::cerr << "service mode requires MULTIMODAL_ENABLE_SERVICE_INTEGRATION=ON\n";
        return 2;
    }
#endif
    const auto backend = complete_visual ? multimodal::VisionComputeBackend::Cblas
                                         : multimodal::VisionComputeBackend::ScalarReference;
    if (!multimodal::vision_compute_backend_available(backend)) {
        std::cerr << "complete visual modes require MULTIMODAL_ENABLE_CBLAS=ON\n";
        return 2;
    }
    std::size_t decoder_layers = 1;
    if (argc >= 5) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(argv[4], &end, 10);
        if (end == argv[4] || *end != '\0' || parsed == 0) {
            std::cerr << "decoder-layers must be a positive integer\n";
            return 2;
        }
        decoder_layers = static_cast<std::size_t>(parsed);
    }
    if (service_mode && decoder_layers != 32) {
        std::cerr << "service mode requires the fixed 32-layer golden\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    multimodal::VisionModelConfig config;
    multimodal::VisionTensorLayout layout;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error) ||
        !multimodal::MmprojSchema::config_from_gguf(index.metadata, &config, &error) ||
        !multimodal::MmprojSchema::validate_gguf(config, index, &layout, &error)) {
        std::cerr << "mmproj failed: " << error << '\n';
        return 1;
    }
    if (config.image_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        config.patch_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        config.merge_size != 4 || config.image_size / config.patch_size < config.merge_size) {
        std::cerr << "mmproj failed: unsupported image, patch, or merge size\n";
        return 1;
    }

    multimodal::PatchEmbeddingWeights embedding_weights;
    multimodal::LayerNormWeights post_norm;
    multimodal::LinearWeights projector_weights;
    if (!multimodal::load_patch_embedding_weights(
            argv[1], index, config, layout, &embedding_weights, &error) ||
        !multimodal::load_post_layer_norm_weights(
            argv[1], index, config, layout, &post_norm, &error) ||
        !multimodal::load_idefics3_projector_weights(
            argv[1], index, config, layout, &projector_weights, &error)) {
        std::cerr << "weight load failed: " << error << '\n';
        return 1;
    }

    multimodal::RgbImage image;
    multimodal::MultimodalStageRecorder metrics_recorder;
    if (!metrics_recorder.begin(
            multimodal::MultimodalRequestStage::ImageDecode, 0, &error) ||
        !multimodal::load_rgb_image(argv[2], &image, &error)) {
        std::cerr << "image load failed: " << error << '\n';
        return 1;
    }
    if (!metrics_recorder.finish(
            multimodal::MultimodalRequestStage::ImageDecode, rgb_bytes(image), &error)) {
        std::cerr << "image metrics failed: " << error << '\n';
        return 1;
    }
    const multimodal::ImagePreprocessorConfig preprocess{
        static_cast<int>(config.image_size), static_cast<int>(config.image_size),
        static_cast<int>(config.patch_size), static_cast<int>(config.patch_size),
        config.image_mean, config.image_std};

    std::size_t tile_rows = 0;
    std::size_t tile_cols = 0;
    std::vector<multimodal::RgbImage> visual_images;
    if (dynamic_tiles) {
        if (!metrics_recorder.begin(
                multimodal::MultimodalRequestStage::DynamicTiling, rgb_bytes(image), &error)) {
            std::cerr << "tiling metrics failed: " << error << '\n';
            return 1;
        }
        const multimodal::Idefics3TilingConfig tiling_config{
            static_cast<int>(config.image_size), static_cast<int>(config.patch_size),
            static_cast<int>(config.merge_size), 256, 256};
        multimodal::Idefics3TilingResult tiling;
        if (!multimodal::prepare_idefics3_tiles(image, tiling_config, &tiling, &error)) {
            std::cerr << "dynamic tiling failed: " << error << '\n';
            return 1;
        }
        tile_rows = static_cast<std::size_t>(tiling.layout.grid_y);
        tile_cols = static_cast<std::size_t>(tiling.layout.grid_x);
        visual_images.reserve(tiling.tiles.size() + 1);
        for (auto& tile : tiling.tiles) visual_images.push_back(std::move(tile.image));
        visual_images.push_back(std::move(tiling.overview.image));
        std::size_t tiled_bytes = 0;
        for (const auto& visual : visual_images) tiled_bytes += rgb_bytes(visual);
        if (!metrics_recorder.finish(
                multimodal::MultimodalRequestStage::DynamicTiling, tiled_bytes, &error)) {
            std::cerr << "tiling metrics failed: " << error << '\n';
            return 1;
        }
        std::cout << "source_image=" << image.width << 'x' << image.height << '\n'
                  << "tile_grid=" << tile_rows << 'x' << tile_cols << '\n'
                  << "visual_count=" << visual_images.size() << '\n';
    } else {
        visual_images.push_back(std::move(image));
    }

    const std::size_t source_grid = config.image_size / config.patch_size;
    const std::size_t selected_grid = complete_visual ? source_grid : config.merge_size;
    const std::string mmproj_path = argv[1];
    const multimodal::SiglipBlockWeightProvider provider =
        [&mmproj_path, &index, &config, &layout](
            std::size_t block, multimodal::SiglipBlockWeights* weights, std::string* message) {
            return multimodal::load_siglip_block_weights(
                mmproj_path, index, config, layout, block, weights, message);
        };
    std::vector<std::size_t> positions;
    positions.reserve(selected_grid * selected_grid);
    for (std::size_t y = 0; y < selected_grid; ++y) {
        for (std::size_t x = 0; x < selected_grid; ++x) {
            positions.push_back(y * source_grid + x);
        }
    }

    std::vector<multimodal::FloatMatrix> visual_embeddings;
    visual_embeddings.reserve(visual_images.size());
    multimodal::FloatMatrix first_encoded;
    multimodal::Idefics3ProjectorTrace first_trace;
    for (std::size_t visual_index = 0; visual_index < visual_images.size(); ++visual_index) {
        if (!metrics_recorder.begin(
                multimodal::MultimodalRequestStage::VisionEncoder,
                rgb_bytes(visual_images[visual_index]) + visual_output_bytes(visual_embeddings),
                &error)) {
            std::cerr << "vision metrics failed: " << error << '\n';
            return 1;
        }
        multimodal::PreprocessedImage prepared;
        if (!multimodal::preprocess_image(
                visual_images[visual_index], preprocess, &prepared, &error)) {
            std::cerr << "preprocess failed for visual " << visual_index << ": " << error << '\n';
            return 1;
        }
        multimodal::PatchTensor selected;
        selected.patch_count = complete_visual
                                   ? prepared.patches.patch_count
                                   : static_cast<int>(config.merge_size * config.merge_size);
        selected.patch_values = prepared.patches.patch_values;
        selected.values.reserve(static_cast<std::size_t>(selected.patch_count) *
                                static_cast<std::size_t>(selected.patch_values));
        for (const std::size_t position : positions) {
            const auto begin = prepared.patches.values.begin() +
                               position * static_cast<std::size_t>(selected.patch_values);
            selected.values.insert(selected.values.end(), begin, begin + selected.patch_values);
        }

        multimodal::FloatMatrix embedding;
        if (!multimodal::embed_patches_at_positions_with_backend(
                selected, positions, embedding_weights, backend, &embedding, &error)) {
            std::cerr << "embedding failed for visual " << visual_index << ": " << error << '\n';
            return 1;
        }
        multimodal::FloatMatrix encoded;
        std::vector<multimodal::SiglipLayerSummary> summaries;
        multimodal::SiglipEncoderStats stats;
        if (!multimodal::run_siglip_encoder_with_backend(
                embedding, config.block_count, config.attention_head_count,
                config.layer_norm_epsilon, provider, post_norm, backend,
                &encoded, &summaries, &stats, &error)) {
            std::cerr << "vision encoder failed for visual " << visual_index
                      << ": " << error << '\n';
            return 1;
        }
        const std::size_t vision_live_bytes =
            prepared.normalized.values.size() * sizeof(float) +
            prepared.patches.values.size() * sizeof(float) +
            selected.values.size() * sizeof(float) + matrix_bytes(embedding) +
            matrix_bytes(encoded) + visual_output_bytes(visual_embeddings);
        if (!metrics_recorder.finish(
                multimodal::MultimodalRequestStage::VisionEncoder,
                vision_live_bytes, &error) ||
            !metrics_recorder.begin(
                multimodal::MultimodalRequestStage::Projector,
                matrix_bytes(encoded) + visual_output_bytes(visual_embeddings), &error)) {
            std::cerr << "vision metrics failed: " << error << '\n';
            return 1;
        }
        multimodal::Idefics3ProjectorTrace trace;
        if (!multimodal::run_idefics3_projector_with_backend(
                encoded, selected_grid, selected_grid, config.merge_size,
                projector_weights, backend, &trace, &error)) {
            std::cerr << "projector failed for visual " << visual_index << ": " << error << '\n';
            return 1;
        }
        if (!metrics_recorder.finish(
                multimodal::MultimodalRequestStage::Projector,
                matrix_bytes(encoded) + matrix_bytes(trace.merged) +
                    matrix_bytes(trace.output) + visual_output_bytes(visual_embeddings),
                &error)) {
            std::cerr << "projector metrics failed: " << error << '\n';
            return 1;
        }
        if (visual_index == 0) {
            first_encoded = encoded;
            first_trace = trace;
        }
        visual_embeddings.push_back(std::move(trace.output));
    }

    std::cout << std::setprecision(10) << "vision_mode="
              << (dynamic_tiles ? "dynamic-tiles" :
                  (complete_visual ? "full-tile" : "top-left-window")) << '\n'
              << "vision_backend=" << (complete_visual ? "cblas" : "scalar") << '\n'
              << "selected_positions=";
    if (complete_visual) {
        std::cout << positions.front() << ".." << positions.back();
    } else {
        for (std::size_t i = 0; i < positions.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << positions[i];
        }
    }
    std::cout << '\n'
              << "encoder_shape=" << first_encoded.rows << 'x' << first_encoded.cols << '\n'
              << "encoder_checksum=" << checksum(first_encoded) << '\n'
              << "merged_shape=" << first_trace.merged.rows << 'x' << first_trace.merged.cols << '\n'
              << "merged_checksum=" << checksum(first_trace.merged) << '\n'
              << "merged_first=" << first_trace.merged.values.front() << '\n'
              << "projector_weight_bytes=" << projector_weights.weight.size() * sizeof(float) << '\n'
              << "output_shape=" << visual_embeddings.front().rows << 'x'
              << visual_embeddings.front().cols << '\n'
              << "output_checksum=" << checksum(visual_embeddings.front()) << '\n'
              << "first=" << visual_embeddings.front().values.front() << '\n'
              << "visual_output_checksums=";
    for (std::size_t i = 0; i < visual_embeddings.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << checksum(visual_embeddings[i]);
    }
    std::cout << '\n' << "projector=q8_0-valid\n";

    if (argc >= 4) {
        multimodal::FloatMatrix text_embeddings;
        chapter01::ModelMetadata language_metadata;
        if (!load_boundary_text_embeddings(
                argv[3], config.projection_dim, &text_embeddings, &language_metadata, &error)) {
            std::cerr << "language embedding failed: " << error << '\n';
            return 1;
        }
        auto decoder_model = chapter01::RealLlamaCpuModel::load(
            argv[3], decoder_layers, &error);
        if (decoder_model == nullptr) {
            std::cerr << "decoder model failed: " << error << '\n';
            return 1;
        }

        chapter01::TokenSequence prompt_tokens;
        std::size_t prompt_prefix_tokens = 0;
        std::size_t prompt_suffix_tokens = 0;
        std::vector<std::size_t> placeholder_positions{1};
        const bool use_real_prompt = argc >= 6;
        if (use_real_prompt) {
#ifdef CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER
            if (!metrics_recorder.begin(
                    multimodal::MultimodalRequestStage::PromptTokenization,
                    visual_output_bytes(visual_embeddings), &error)) {
                std::cerr << "prompt metrics failed: " << error << '\n';
                return 1;
            }
            multimodal::SmolvlmPromptChunks prompt;
            if (!multimodal::build_smolvlm_single_turn_prompt(
                    argv[5], tile_rows, tile_cols, &prompt, &error) ||
                prompt.visual_count != visual_embeddings.size()) {
                std::cerr << "prompt assembly failed: " << error << '\n';
                return 1;
            }
            const auto tokenizer = chapter01::RealTokenizer::load(argv[3], &error);
            if (tokenizer == nullptr) {
                std::cerr << "tokenizer load failed: " << error << '\n';
                return 1;
            }
            placeholder_positions.clear();
            for (const auto& chunk : prompt.chunks) {
                if (chunk.kind == multimodal::PromptChunkKind::Text) {
                    const auto tokens = tokenizer->encode(chunk.text, false, true);
                    prompt_tokens.insert(prompt_tokens.end(), tokens.begin(), tokens.end());
                } else {
                    if (chunk.visual_index != placeholder_positions.size()) {
                        std::cerr << "prompt visual indexes are not contiguous\n";
                        return 1;
                    }
                    placeholder_positions.push_back(prompt_tokens.size());
                    prompt_tokens.push_back(language_metadata.bos_token_id);
                }
            }
            prompt_prefix_tokens = placeholder_positions.front();
            prompt_suffix_tokens =
                prompt_tokens.size() - placeholder_positions.back() - 1;
            cpp_llm_engine::core::DecoderEmbeddingSequence embedded;
            if (!decoder_model->embed_tokens(prompt_tokens, &embedded, &error)) {
                std::cerr << "prompt embedding failed: " << error << '\n';
                return 1;
            }
            text_embeddings = {
                embedded.token_count, embedded.hidden_size, std::move(embedded.values)};
            if (!metrics_recorder.finish(
                    multimodal::MultimodalRequestStage::PromptTokenization,
                    visual_output_bytes(visual_embeddings) + matrix_bytes(text_embeddings),
                    &error)) {
                std::cerr << "prompt metrics failed: " << error << '\n';
                return 1;
            }
#else
            std::cerr << "real prompt requires MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER=ON\n";
            return 1;
#endif
        }

        std::vector<multimodal::VisualEmbeddingInsertion> insertions;
        insertions.reserve(visual_embeddings.size());
        for (std::size_t i = 0; i < visual_embeddings.size(); ++i) {
            insertions.push_back({placeholder_positions[i], visual_embeddings[i]});
        }
        multimodal::MultimodalEmbeddingSequence sequence;
        if (!metrics_recorder.begin(
                multimodal::MultimodalRequestStage::EmbeddingComposition,
                matrix_bytes(text_embeddings) + visual_output_bytes(visual_embeddings),
                &error) ||
            !multimodal::compose_multimodal_embeddings(
                text_embeddings, insertions, &sequence, &error)) {
            std::cerr << "multimodal input failed: " << error << '\n';
            return 1;
        }
        if (!metrics_recorder.finish(
                multimodal::MultimodalRequestStage::EmbeddingComposition,
                matrix_bytes(text_embeddings) + visual_output_bytes(visual_embeddings) +
                    matrix_bytes(sequence.embeddings),
                &error) ||
            !metrics_recorder.set_sequence_shape(
                visual_embeddings.size(),
                visual_embeddings.size() * visual_embeddings.front().rows,
                sequence.embeddings.rows, &error)) {
            std::cerr << "multimodal metrics failed: " << error << '\n';
            return 1;
        }
        std::cout << "language_architecture=" << language_metadata.architecture << '\n'
                  << "language_hidden=" << language_metadata.hidden_size << '\n'
                  << "text_token_ids=" << language_metadata.bos_token_id << ','
                  << language_metadata.eos_token_id << '\n'
                  << "text_embedding_checksum=" << checksum(text_embeddings) << '\n'
                  << "sequence_shape=" << sequence.embeddings.rows << 'x'
                  << sequence.embeddings.cols << '\n'
                  << "sequence_checksum=" << checksum(sequence.embeddings) << '\n'
                  << "multimodal_input=q8_0-valid\n";
        if (use_real_prompt) {
            std::cout << "prompt_mode="
                      << (dynamic_tiles ? "smolvlm-dynamic-tiles" :
                          (complete_visual ? "smolvlm-single-tile" :
                                             "smolvlm-overview-only")) << '\n'
                      << "prompt_prefix_tokens=" << prompt_prefix_tokens << '\n'
                      << "prompt_suffix_tokens=" << prompt_suffix_tokens << '\n'
                      << "prompt_visual_count=" << visual_embeddings.size() << '\n'
                      << "prompt_visual_rows="
                      << visual_embeddings.size() * visual_embeddings.front().rows << '\n'
                      << "prompt_token_ids=";
            for (std::size_t index = 0; index < prompt_tokens.size(); ++index) {
                if (index != 0) std::cout << ',';
                if (std::find(placeholder_positions.begin(), placeholder_positions.end(), index) !=
                    placeholder_positions.end()) {
                    std::cout << "visual";
                } else {
                    std::cout << prompt_tokens[index];
                }
            }
            std::cout << '\n';
        } else {
            std::cout << "sequence_origins=text:0,visual:0:0,text:2\n";
        }
#ifdef CPP_LLM_MULTIMODAL_HAS_SERVICE
        if (service_mode) {
            chapter06_generation::GenerationConfig generation_config;
            generation_config.max_new_tokens = 1;
            generation_config.max_context_tokens =
                decoder_model->decoder_contract().max_context_tokens;
            generation_config.eos_token_id = language_metadata.eos_token_id;
            chapter06_generation::Runtime runtime(generation_config);
            chapter06_generation::GreedySampler sampler;
            chapter10_serving::ServiceCore service(2, 2);
            std::size_t factory_calls = 0;
            chapter10_serving::HttpAdapter adapter(
                service,
                runtime,
                sampler,
                [&]() -> std::shared_ptr<chapter06_generation::RequestModelExecutor> {
                    ++factory_calls;
                    cpp_llm_engine::core::DecoderEmbeddingSequence input{
                        sequence.embeddings.rows,
                        sequence.embeddings.cols,
                        sequence.embeddings.values,
                    };
                    std::string create_error;
                    return cpp_llm_engine::EmbeddingRequestExecutor::create(
                        decoder_model->create_context(), std::move(input), &create_error);
                });
            if (!metrics_recorder.begin(
                    multimodal::MultimodalRequestStage::LanguagePrefill,
                    matrix_bytes(sequence.embeddings), &error)) {
                std::cerr << "language metrics failed: " << error << '\n';
                return 1;
            }
            const std::string sync_body =
                service_request_body(701, sequence.embeddings.rows);
            const auto sync_response = adapter.handle(
                {"POST", "/v1/completions", sync_body});
            RecordingWriter writer;
            const std::string stream_body =
                service_request_body(702, sequence.embeddings.rows);
            const auto stream_response = adapter.handle_stream(
                {"POST", "/v1/completions", stream_body}, writer);
            if (!metrics_recorder.finish(
                    multimodal::MultimodalRequestStage::LanguagePrefill,
                    matrix_bytes(sequence.embeddings), &error)) {
                std::cerr << "language metrics failed: " << error << '\n';
                return 1;
            }
            const std::string sync_token = "\"generated_token_ids\":[17600]";
            const std::string stream_token = "data: {\"token\":17600,\"step\":0}";
            const auto& service_metrics = service.metrics();
            const bool sync_match = sync_response.status_code == 200 &&
                sync_response.body.find(sync_token) != std::string::npos;
            const bool stream_match = stream_response.status_code == 200 &&
                writer.head.status_code == 200 && writer.finished &&
                writer.body.find(stream_token) != std::string::npos &&
                writer.body.find("finish_reason\":\"stop") != std::string::npos;
            const bool metrics_match = factory_calls == 2 &&
                service_metrics.accepted_total == 2 &&
                service_metrics.completed_total == 2 &&
                service_metrics.released_total == 2 &&
                service_metrics.generated_tokens_total == 2 &&
                service.inflight_size() == 0;
            std::cout << "service_sync_status=" << sync_response.status_code << '\n'
                      << "service_sync_token_match=" << std::boolalpha << sync_match << '\n'
                      << "service_sse_status=" << stream_response.status_code << '\n'
                      << "service_sse_token_match=" << stream_match << '\n'
                      << "service_factory_calls=" << factory_calls << '\n'
                      << "service_accepted=" << service_metrics.accepted_total << '\n'
                      << "service_completed=" << service_metrics.completed_total << '\n'
                      << "service_released=" << service_metrics.released_total << '\n'
                      << "service_generated_tokens="
                      << service_metrics.generated_tokens_total << '\n'
                      << "service_inflight=" << service.inflight_size() << '\n'
                      << "multimodal_service="
                      << (sync_match && stream_match && metrics_match ? "valid" : "invalid")
                      << '\n';
            if (!sync_match || !stream_match || !metrics_match) return 1;
        } else
#endif
        {
            auto decoder_context = decoder_model->create_context();
            cpp_llm_engine::core::DecoderEmbeddingSequence decoder_input{
                sequence.embeddings.rows,
                sequence.embeddings.cols,
                sequence.embeddings.values,
            };
            cpp_llm_engine::core::DecoderLogits logits;
            if (!metrics_recorder.begin(
                    multimodal::MultimodalRequestStage::LanguagePrefill,
                    matrix_bytes(sequence.embeddings), &error)) {
                std::cerr << "language metrics failed: " << error << '\n';
                return 1;
            }
            const auto decoder_status = decoder_context->prefill_embeddings(decoder_input, &logits);
            if (!decoder_status.is_ok()) {
                std::cerr << "decoder embedding prefill failed: "
                          << decoder_status.message() << '\n';
                return 1;
            }
            if (!metrics_recorder.finish(
                    multimodal::MultimodalRequestStage::LanguagePrefill,
                    matrix_bytes(sequence.embeddings) + logits.size() * sizeof(float),
                    &error)) {
                std::cerr << "language metrics failed: " << error << '\n';
                return 1;
            }
            const auto top = std::max_element(logits.begin(), logits.end());
            double logits_checksum = 0.0;
            for (std::size_t i = 0; i < logits.size(); ++i) {
                logits_checksum += static_cast<double>((i % 97) + 1) * logits[i];
            }
            std::cout << "decoder_layers=" << decoder_layers << '\n'
                      << "decoder_tokens=" << decoder_context->token_count() << '\n'
                      << "logits_size=" << logits.size() << '\n'
                      << "logits_checksum=" << logits_checksum << '\n'
                      << "top1_token_id=" << std::distance(logits.begin(), top) << '\n'
                      << "top1_logit=" << *top << '\n'
                      << "multimodal_decoder=q8_0-" << decoder_layers
                      << "-layer-valid\n";
        }
        const auto& request_metrics = metrics_recorder.metrics();
        for (std::size_t stage = 0;
             stage < static_cast<std::size_t>(multimodal::MultimodalRequestStage::Count);
             ++stage) {
            const auto value = static_cast<multimodal::MultimodalRequestStage>(stage);
            const auto& stage_metrics = request_metrics.stage(value);
            std::cout << "stage_" << multimodal::multimodal_request_stage_name(value)
                      << "_seconds=" << stage_metrics.elapsed_seconds << '\n'
                      << "stage_" << multimodal::multimodal_request_stage_name(value)
                      << "_count=" << stage_metrics.invocation_count << '\n'
                      << "stage_" << multimodal::multimodal_request_stage_name(value)
                      << "_peak_live_bytes=" << stage_metrics.peak_live_bytes << '\n';
        }
        std::cout << "input_pipeline_seconds=" << request_metrics.input_pipeline_seconds() << '\n'
                  << "request_peak_live_bytes=" << request_metrics.peak_live_bytes << '\n';
    }
    return 0;
}
