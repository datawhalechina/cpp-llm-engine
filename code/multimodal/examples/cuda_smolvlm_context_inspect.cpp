// 多模态 CUDA：真实视觉 embedding 接入 SmolVLM CUDA 语言 Context。
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"
#include "cpp_llm_engine/model/real_llama_cuda_model.h"
#include "multimodal/cuda_vision_pipeline.h"
#include "multimodal/image.h"
#include "multimodal/image_preprocessor.h"
#include "multimodal/http_request.h"
#include "multimodal/input_embedding.h"
#include "multimodal/mmproj_schema.h"
#include "multimodal/smolvlm_prompt.h"

#ifdef CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/cuda_embedding_batch_runner.h"
#include "cpp_llm_engine/embedding_request_executor.h"
#include "real_tokenizer.h"
#endif
#ifdef CPP_LLM_MULTIMODAL_HAS_SERVICE
#include "chapter10_serving/http_adapter.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

#ifdef CPP_LLM_MULTIMODAL_HAS_SERVICE
class RecordingWriter final : public chapter10_serving::ResponseWriter {
public:
    bool write_headers(const chapter10_serving::ResponseHead& value) override {
        head = value;
        headers_written = true;
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
    bool headers_written = false;
    bool finished = false;
};

std::string base64_encode(const std::vector<std::uint8_t>& bytes) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((bytes.size() + 2) / 3 * 4);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3) {
        const std::uint32_t first = bytes[offset];
        const std::uint32_t second =
            offset + 1 < bytes.size() ? bytes[offset + 1] : 0;
        const std::uint32_t third =
            offset + 2 < bytes.size() ? bytes[offset + 2] : 0;
        const std::uint32_t bits = (first << 16U) | (second << 8U) | third;
        encoded.push_back(kAlphabet[(bits >> 18U) & 0x3fU]);
        encoded.push_back(kAlphabet[(bits >> 12U) & 0x3fU]);
        encoded.push_back(
            offset + 1 < bytes.size() ? kAlphabet[(bits >> 6U) & 0x3fU] : '=');
        encoded.push_back(
            offset + 2 < bytes.size() ? kAlphabet[bits & 0x3fU] : '=');
    }
    return encoded;
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    for (const char ch : value) {
        switch (ch) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped.push_back(ch); break;
        }
    }
    return escaped;
}

std::string service_request_body(
    std::size_t request_id,
    const multimodal::RgbImage& image,
    const std::string& prompt) {
    std::ostringstream ppm;
    ppm << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    const std::string header = ppm.str();
    std::vector<std::uint8_t> bytes(header.begin(), header.end());
    bytes.insert(bytes.end(), image.pixels.begin(), image.pixels.end());
    std::ostringstream body;
    body << "{\"request_id\":" << request_id
         << ",\"prompt\":\"" << json_escape(prompt)
         << "\",\"image_media_type\":\"image/ppm\",\"image_base64\":\""
         << base64_encode(bytes) << "\"}";
    return body.str();
}

std::string token_array_json(const chapter01::TokenSequence& tokens) {
    std::ostringstream output;
    output << "\"generated_token_ids\":[";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) output << ',';
        output << tokens[index];
    }
    output << ']';
    return output.str();
}

bool contains_tokens_in_order(
    const std::string& text,
    const chapter01::TokenSequence& tokens) {
    std::size_t position = 0;
    for (const auto token : tokens) {
        const std::string marker = "\"token\":" + std::to_string(token);
        position = text.find(marker, position);
        if (position == std::string::npos) return false;
        position += marker.size();
    }
    return true;
}

#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5 && argc != 6) {
        std::cerr << "usage: multimodal_cuda_smolvlm_context_inspect "
                     "<mmproj.gguf> <model.gguf> <image> "
                     "[--full-tile | --mixed-prompt <question> | "
                     "--mixed-service <question>]\n";
        return 2;
    }
    const bool service_mode = argc == 6 &&
        std::string(argv[4]) == "--mixed-service";
    const bool mixed_prompt = argc == 6 &&
        (std::string(argv[4]) == "--mixed-prompt" || service_mode);
    const bool full_tile = mixed_prompt ||
        (argc == 5 && std::string(argv[4]) == "--full-tile");
    if ((argc == 5 && !full_tile) || (argc == 6 && !mixed_prompt)) {
        std::cerr << "unknown SmolVLM context option: " << argv[4] << '\n';
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
    const std::size_t selected_grid =
        full_tile ? source_grid : config.merge_size;
    const std::size_t visual_tokens =
        (selected_grid / config.merge_size) *
        (selected_grid / config.merge_size);
    multimodal::PatchTensor selected;
    selected.patch_count = static_cast<int>(selected_grid * selected_grid);
    selected.patch_values = prepared.patches.patch_values;
    selected.values.reserve(
        static_cast<std::size_t>(selected.patch_count) *
        static_cast<std::size_t>(selected.patch_values));
    std::vector<std::size_t> positions;
    positions.reserve(static_cast<std::size_t>(selected.patch_count));
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
        {visual_tokens, config.projection_dim}, DType::Float32, &visual_output);
    if (status.is_ok()) status = pipeline.forward(
        &vision_context, selected, positions, selected_grid, selected_grid,
        &visual_output);
    std::vector<float> visual_embedding(
        visual_tokens * config.projection_dim);
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
    cpp_llm_engine::core::DecoderEmbeddingSequence embeddings{
        visual_tokens, config.projection_dim, visual_embedding};
    std::size_t text_tokens = 0;
    std::size_t prompt_prefix_tokens = 0;
    std::size_t prompt_suffix_tokens = 0;
#ifdef CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER
    std::unique_ptr<chapter01::RealTokenizer> tokenizer;
#endif
    if (mixed_prompt) {
#ifdef CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER
        multimodal::SmolvlmPromptChunks prompt;
        if (!multimodal::build_smolvlm_single_turn_prompt(
                argv[5], 0, 0, &prompt, &error) || prompt.visual_count != 1) {
            std::cerr << "SmolVLM prompt assembly failed: " << error << '\n';
            return 1;
        }
        tokenizer = chapter01::RealTokenizer::load(model_path, &error);
        if (tokenizer == nullptr) {
            std::cerr << "SmolVLM tokenizer load failed: " << error << '\n';
            return 1;
        }
        chapter01::TokenSequence prompt_tokens;
        std::vector<std::size_t> placeholder_positions;
        for (const auto& chunk : prompt.chunks) {
            if (chunk.kind == multimodal::PromptChunkKind::Text) {
                const auto tokens = tokenizer->encode(chunk.text, false, true);
                prompt_tokens.insert(
                    prompt_tokens.end(), tokens.begin(), tokens.end());
            } else {
                placeholder_positions.push_back(prompt_tokens.size());
                prompt_tokens.push_back(0);
            }
        }
        if (placeholder_positions.size() != 1) {
            std::cerr << "SmolVLM mixed prompt requires one visual placeholder\n";
            return 1;
        }
        prompt_prefix_tokens = placeholder_positions.front();
        prompt_suffix_tokens =
            prompt_tokens.size() - placeholder_positions.front() - 1;
        text_tokens = prompt_tokens.size() - 1;
        cpp_llm_engine::core::DecoderEmbeddingSequence text_embeddings;
        if (!cpu_model->embed_tokens(prompt_tokens, &text_embeddings, &error)) {
            std::cerr << "SmolVLM text embedding failed: " << error << '\n';
            return 1;
        }
        multimodal::MultimodalEmbeddingSequence sequence;
        const multimodal::FloatMatrix text_matrix{
            text_embeddings.token_count,
            text_embeddings.hidden_size,
            std::move(text_embeddings.values)};
        const multimodal::FloatMatrix visual_matrix{
            visual_tokens, config.projection_dim, std::move(visual_embedding)};
        if (!multimodal::compose_multimodal_embeddings(
                text_matrix,
                {{placeholder_positions.front(), visual_matrix}},
                &sequence,
                &error)) {
            std::cerr << "SmolVLM mixed embedding assembly failed: " << error << '\n';
            return 1;
        }
        embeddings = {
            sequence.embeddings.rows,
            sequence.embeddings.cols,
            std::move(sequence.embeddings.values)};
#else
        std::cerr << "mixed prompt requires "
                     "MULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER=ON\n";
        return 1;
#endif
    }

    auto cpu_context = cpu_model->create_context();
    auto cuda_context = cuda_model->create_context(embeddings.token_count + 4, &error);
    if (cpu_context == nullptr || cuda_context == nullptr) {
        std::cerr << "SmolVLM context creation failed: " << error << '\n';
        return 1;
    }
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
    bool matched = cpu_logits.size() == cuda_logits.size() &&
        std::isfinite(maximum_error) && std::isfinite(cosine) &&
        cosine >= 0.999 && cpu_token == cuda_token &&
        cpu_context->token_count() == embeddings.token_count &&
        cuda_context->position() == embeddings.token_count;
    float decode_maximum_error = 0.0F;
    double decode_cosine = 1.0;
    std::ptrdiff_t cpu_decode_token = -1;
    std::ptrdiff_t cuda_decode_token = -1;
    chapter01::TokenSequence runtime_cpu_tokens;
    chapter01::TokenSequence runtime_cuda_tokens;
    std::string runtime_generated_text;
    std::size_t runtime_cpu_position = 0;
    std::size_t runtime_cuda_position = 0;
    bool service_sync_match = false;
    bool service_stream_match = false;
    bool service_metrics_match = false;
    std::size_t service_factory_calls = 0;
    bool service_batch_match = false;
    bool service_batch_released = false;
    std::size_t service_model_batch_calls = 0;
    std::size_t service_tensor_batch_requests = 0;
    std::size_t service_prefill_group_calls = 0;
    std::size_t service_prefill_group_requests = 0;
    std::size_t service_batch_workspaces = 0;
    std::size_t service_vision_batch_calls = 0;
    std::size_t service_vision_max_batch_size = 0;
    std::size_t service_async_prefill_group_calls = 0;
    std::size_t service_async_tensor_batch_requests = 0;
    std::size_t service_async_packed_prefill_calls = 0;
    std::size_t service_packed_prefill_calls = 0;
    std::size_t service_prefill_workspaces = 0;
    double service_serial_prefill_ms = 0.0;
    double service_batched_prefill_ms = 0.0;
    if (mixed_prompt && matched) {
        std::vector<float> cpu_decode_logits;
        std::vector<float> cuda_decode_logits;
        const auto cpu_decode_status = cpu_context->decode_token(
            static_cast<std::int32_t>(cpu_token), &cpu_decode_logits);
        const auto cuda_decode_status = cuda_context->forward_token(
            static_cast<std::int32_t>(cuda_token), &cuda_decode_logits);
        if (!cpu_decode_status.is_ok() || !cuda_decode_status.is_ok() ||
            cpu_decode_logits.size() != cuda_decode_logits.size()) {
            std::cerr << "SmolVLM mixed decode failed: "
                      << (cpu_decode_status.is_ok()
                              ? cuda_decode_status.message()
                              : cpu_decode_status.message())
                      << '\n';
            return 1;
        }
        double decode_dot = 0.0;
        double cpu_decode_square = 0.0;
        double cuda_decode_square = 0.0;
        for (std::size_t value = 0; value < cpu_decode_logits.size(); ++value) {
            decode_maximum_error = std::max(
                decode_maximum_error,
                std::fabs(cpu_decode_logits[value] - cuda_decode_logits[value]));
            decode_dot += static_cast<double>(cpu_decode_logits[value]) *
                cuda_decode_logits[value];
            cpu_decode_square += static_cast<double>(cpu_decode_logits[value]) *
                cpu_decode_logits[value];
            cuda_decode_square += static_cast<double>(cuda_decode_logits[value]) *
                cuda_decode_logits[value];
        }
        decode_cosine = decode_dot /
            std::sqrt(cpu_decode_square * cuda_decode_square);
        cpu_decode_token = std::distance(
            cpu_decode_logits.begin(),
            std::max_element(cpu_decode_logits.begin(), cpu_decode_logits.end()));
        cuda_decode_token = std::distance(
            cuda_decode_logits.begin(),
            std::max_element(cuda_decode_logits.begin(), cuda_decode_logits.end()));
        matched = std::isfinite(decode_maximum_error) &&
            std::isfinite(decode_cosine) && decode_cosine >= 0.999 &&
            cpu_decode_token == cuda_decode_token &&
            cpu_context->token_count() == embeddings.token_count + 1 &&
            cuda_context->position() == embeddings.token_count + 1;
#ifdef CPP_LLM_MULTIMODAL_HAS_REAL_TOKENIZER
        auto cpu_runtime_context = cpu_model->create_context();
        auto cuda_runtime_context = cuda_model->create_context(
            embeddings.token_count + 4, &error);
        auto cpu_executor = cpp_llm_engine::EmbeddingRequestExecutor::create(
            std::move(cpu_runtime_context), embeddings, &error);
        auto cuda_executor = cpp_llm_engine::EmbeddingRequestExecutor::create(
            std::move(cuda_runtime_context), embeddings, &error);
        if (cpu_executor == nullptr || cuda_executor == nullptr) {
            std::cerr << "SmolVLM Runtime executor creation failed: "
                      << error << '\n';
            return 1;
        }
        chapter06_generation::GenerationConfig generation;
        generation.max_new_tokens = 4;
        generation.max_context_tokens = embeddings.token_count + 4;
        generation.eos_token_id = cuda_model->eos_token_id();
        chapter06_generation::Runtime runtime(generation);
        chapter06_generation::GreedySampler sampler;
        const chapter06_generation::TokenSequence positions(
            embeddings.token_count, 0);
        chapter06_generation::Request cpu_request(positions);
        chapter06_generation::Request cuda_request(positions);
        const std::size_t cpu_steps = runtime.generate(
            cpu_request, sampler, *cpu_executor);
        const std::size_t cuda_steps = runtime.generate(
            cuda_request, sampler, *cuda_executor);
        runtime_cpu_tokens.assign(
            cpu_request.generated().begin(), cpu_request.generated().end());
        runtime_cuda_tokens.assign(
            cuda_request.generated().begin(), cuda_request.generated().end());
        runtime_cpu_position = cpu_executor->context_token_count();
        runtime_cuda_position = cuda_executor->context_token_count();
        runtime_generated_text = tokenizer->decode(
            runtime_cuda_tokens, false, true);
        matched = matched && cpu_steps == 4 && cuda_steps == 4 &&
            cpu_request.finished() && cuda_request.finished() &&
            runtime_cpu_tokens == runtime_cuda_tokens &&
            runtime_cpu_position == embeddings.token_count + 3 &&
            runtime_cuda_position == embeddings.token_count + 3;
#ifdef CPP_LLM_MULTIMODAL_HAS_SERVICE
        if (service_mode && matched) {
            cpp_llm_engine::CudaEmbeddingBatchRunner service_runner(
                *cuda_model,
                cpp_llm_engine::CudaEmbeddingPrefillPath::PackedEqualLength);
            chapter10_serving::ServiceCore service(
                2, 2, {}, {}, {}, &service_runner);
            chapter10_serving::AsyncServiceLoop async_service(
                service, runtime, sampler, 2048, 0,
                std::chrono::milliseconds(10));
            chapter10_serving::CompletionRequestCollector preparation_collector(
                [&](const std::vector<chapter10_serving::HttpRequest>& requests,
                    std::vector<chapter10_serving::CompletionPreparationResult>* results) {
                    service_factory_calls += requests.size();
                    results->assign(requests.size(), {});
                    std::vector<multimodal::MultimodalHttpRequest> parsed(
                        requests.size());
                    std::vector<multimodal::PreprocessedImage> prepared_images(
                        requests.size());
                    std::vector<std::size_t> valid_indices;
                    std::vector<multimodal::PatchTensor> patch_batch;
                    std::vector<std::vector<std::size_t>> position_batch;
                    for (std::size_t index = 0; index < requests.size(); ++index) {
                        std::string request_error;
                        if (!multimodal::parse_multimodal_http_request(
                                requests[index].body, &parsed[index], &request_error) ||
                            !multimodal::preprocess_image(
                                parsed[index].image, preprocess,
                                &prepared_images[index], &request_error)) {
                            (*results)[index].status =
                                chapter10_serving::CompletionPrepareStatus::invalid_request;
                            (*results)[index].error = std::move(request_error);
                            continue;
                        }
                        std::vector<std::size_t> request_positions(
                            prepared_images[index].patches.patch_count);
                        for (std::size_t position = 0;
                             position < request_positions.size(); ++position) {
                            request_positions[position] = position;
                        }
                        valid_indices.push_back(index);
                        patch_batch.push_back(prepared_images[index].patches);
                        position_batch.push_back(std::move(request_positions));
                    }
                    if (valid_indices.empty()) return;

                    CudaTensor request_visual_output;
                    auto request_status = CudaTensor::create_contiguous(
                        {valid_indices.size() * visual_tokens, config.projection_dim},
                        DType::Float32,
                        &request_visual_output);
                    if (request_status.is_ok()) {
                        request_status = pipeline.forward_batch(
                            &vision_context, patch_batch, position_batch,
                            source_grid, source_grid, &request_visual_output);
                    }
                    std::vector<float> packed_visual_values(
                        valid_indices.size() * visual_tokens * config.projection_dim);
                    if (request_status.is_ok()) {
                        request_status = vision_context.download(
                            request_visual_output,
                            packed_visual_values.data(),
                            packed_visual_values.size() * sizeof(float));
                    }
                    if (request_status.is_ok()) {
                        request_status = vision_context.synchronize();
                    }
                    if (!request_status.is_ok()) {
                        for (const std::size_t index : valid_indices) {
                            (*results)[index].status =
                                chapter10_serving::CompletionPrepareStatus::executor_unavailable;
                            (*results)[index].error = request_status.message();
                        }
                        return;
                    }

                    const std::size_t values_per_image =
                        visual_tokens * config.projection_dim;
                    for (std::size_t slot = 0; slot < valid_indices.size(); ++slot) {
                        const std::size_t index = valid_indices[slot];
                        auto& result = (*results)[index];
                        std::string create_error;
                        multimodal::SmolvlmPromptChunks request_prompt;
                        if (!multimodal::build_smolvlm_single_turn_prompt(
                                parsed[index].prompt, 0, 0,
                                &request_prompt, &create_error)) {
                            result.status = chapter10_serving::
                                CompletionPrepareStatus::invalid_request;
                            result.error = std::move(create_error);
                            continue;
                        }
                        chapter01::TokenSequence request_tokens;
                        std::size_t placeholder =
                            std::numeric_limits<std::size_t>::max();
                        for (const auto& chunk : request_prompt.chunks) {
                            if (chunk.kind == multimodal::PromptChunkKind::Text) {
                                const auto chunk_tokens = tokenizer->encode(
                                    chunk.text, false, true);
                                request_tokens.insert(
                                    request_tokens.end(), chunk_tokens.begin(),
                                    chunk_tokens.end());
                            } else if (placeholder ==
                                       std::numeric_limits<std::size_t>::max()) {
                                placeholder = request_tokens.size();
                                request_tokens.push_back(0);
                            } else {
                                create_error =
                                    "service request contains multiple image placeholders";
                                break;
                            }
                        }
                        cpp_llm_engine::core::DecoderEmbeddingSequence request_text;
                        if (!create_error.empty() ||
                            placeholder == std::numeric_limits<std::size_t>::max() ||
                            !cpu_model->embed_tokens(
                                request_tokens, &request_text, &create_error)) {
                            result.status = chapter10_serving::
                                CompletionPrepareStatus::executor_unavailable;
                            result.error = std::move(create_error);
                            continue;
                        }
                        std::vector<float> request_visual_values(
                            packed_visual_values.begin() + slot * values_per_image,
                            packed_visual_values.begin() + (slot + 1) * values_per_image);
                        multimodal::MultimodalEmbeddingSequence request_sequence;
                        if (!multimodal::compose_multimodal_embeddings(
                                {request_text.token_count, request_text.hidden_size,
                                 std::move(request_text.values)},
                                {{placeholder,
                                  {visual_tokens, config.projection_dim,
                                   std::move(request_visual_values)}}},
                                &request_sequence, &create_error)) {
                            result.status = chapter10_serving::
                                CompletionPrepareStatus::executor_unavailable;
                            result.error = std::move(create_error);
                            continue;
                        }
                        cpp_llm_engine::core::DecoderEmbeddingSequence request_embeddings{
                            request_sequence.embeddings.rows,
                            request_sequence.embeddings.cols,
                            std::move(request_sequence.embeddings.values)};
                        auto context = cuda_model->create_context(
                            request_embeddings.token_count + 4, &create_error);
                        result.prepared.executor =
                            cpp_llm_engine::EmbeddingRequestExecutor::create(
                                std::move(context), request_embeddings, &create_error);
                        if (result.prepared.executor == nullptr) {
                            result.status = chapter10_serving::
                                CompletionPrepareStatus::executor_unavailable;
                            result.error = std::move(create_error);
                            continue;
                        }
                        result.prepared.prompt_positions.assign(
                            request_embeddings.token_count, 0);
                        result.status =
                            chapter10_serving::CompletionPrepareStatus::ready;
                    }
                },
                2,
                std::chrono::milliseconds(10));
            chapter10_serving::HttpAdapter adapter(
                service, runtime, sampler, {}, {}, {},
                &async_service, &preparation_collector);
            const std::string visual_request = service_request_body(
                901, image, argv[5]);
            chapter10_serving::HttpResponse sync_response;
            chapter10_serving::HttpResponse stream_response;
            RecordingWriter writer;
            const std::string stream_request = service_request_body(
                902, image, argv[5]);
            std::thread sync_client([&] {
                sync_response = adapter.handle({
                    "POST", "/v1/completions", visual_request});
            });
            std::thread stream_client([&] {
                stream_response = adapter.handle_stream({
                    "POST", "/v1/completions", stream_request}, writer);
            });
            sync_client.join();
            stream_client.join();
            service_sync_match = sync_response.status_code == 200 &&
                sync_response.body.find(token_array_json(runtime_cuda_tokens)) !=
                    std::string::npos;
            service_stream_match = stream_response.status_code == 200 &&
                writer.headers_written && writer.finished &&
                writer.head.status_code == 200 &&
                writer.head.content_type == "text/event-stream" &&
                contains_tokens_in_order(writer.body, runtime_cuda_tokens) &&
                writer.body.find("\"finish_reason\":\"stop\"") !=
                    std::string::npos;
            const auto service_snapshot = async_service.snapshot();
            const auto& metrics = service_snapshot.metrics;
            const auto collector_stats = preparation_collector.stats();
            service_vision_batch_calls = collector_stats.batch_calls;
            service_vision_max_batch_size = collector_stats.max_batch_size;
            service_async_prefill_group_calls =
                service_snapshot.dispatch.prefill_group_calls;
            service_async_tensor_batch_requests =
                service_snapshot.dispatch.tensor_batch_requests;
            service_async_packed_prefill_calls =
                service_runner.packed_prefill_count();
            service_metrics_match = service_factory_calls == 2 &&
                service_vision_batch_calls == 1 &&
                service_vision_max_batch_size == 2 &&
                service_async_prefill_group_calls == 1 &&
                service_async_packed_prefill_calls == 1 &&
                service_async_tensor_batch_requests == 6 &&
                metrics.accepted_total == 2 && metrics.completed_total == 2 &&
                metrics.released_total == 2 &&
                metrics.generated_tokens_total == 8 &&
                service_snapshot.waiting == 0 && service_snapshot.running == 0;
            auto serial_executor_a = cpp_llm_engine::EmbeddingRequestExecutor::create(
                cuda_model->create_context(embeddings.token_count + 4, &error),
                embeddings,
                &error);
            auto serial_executor_b = cpp_llm_engine::EmbeddingRequestExecutor::create(
                cuda_model->create_context(embeddings.token_count + 4, &error),
                embeddings,
                &error);
            chapter06_generation::Request serial_request_a(positions);
            chapter06_generation::Request serial_request_b(positions);
            chapter10_serving::ServiceCore serial_service(2, 2);
            const bool serial_admitted = serial_executor_a != nullptr &&
                serial_executor_b != nullptr &&
                serial_service.admit(
                    905, serial_request_a, sampler, serial_executor_a).accepted() &&
                serial_service.admit(
                    906, serial_request_b, sampler, serial_executor_b).accepted();
            const auto serial_start = std::chrono::steady_clock::now();
            const auto serial_prefill_stats = serial_service.dispatch_step(runtime, 2048);
            const auto serial_end = std::chrono::steady_clock::now();
            service_serial_prefill_ms = std::chrono::duration<double, std::milli>(
                serial_end - serial_start).count();
            for (std::size_t step = 1;
                 step < 8 &&
                     (!serial_request_a.terminal() || !serial_request_b.terminal());
                 ++step) {
                if (serial_service.dispatch_step(runtime, 2048).scheduled_requests == 0) {
                    break;
                }
            }
            const bool serial_released =
                serial_service.release_terminal() == 2 &&
                serial_service.inflight_size() == 0;
            cpp_llm_engine::CudaEmbeddingBatchRunner batch_runner(
                *cuda_model,
                cpp_llm_engine::CudaEmbeddingPrefillPath::PackedEqualLength);
            auto batch_executor_a = cpp_llm_engine::EmbeddingRequestExecutor::create(
                cuda_model->create_context(embeddings.token_count + 4, &error),
                embeddings,
                &error);
            auto batch_executor_b = cpp_llm_engine::EmbeddingRequestExecutor::create(
                cuda_model->create_context(embeddings.token_count + 4, &error),
                embeddings,
                &error);
            chapter06_generation::Request batch_request_a(positions);
            chapter06_generation::Request batch_request_b(positions);
            chapter10_serving::ServiceCore batch_service(
                2, 2, {}, {}, {}, &batch_runner);
            const bool batch_admitted = batch_executor_a != nullptr &&
                batch_executor_b != nullptr &&
                batch_service.admit(
                    903, batch_request_a, sampler, batch_executor_a).accepted() &&
                batch_service.admit(
                    904, batch_request_b, sampler, batch_executor_b).accepted();
            const auto batch_start = std::chrono::steady_clock::now();
            const auto batch_prefill_stats = batch_service.dispatch_step(runtime, 2048);
            const auto batch_end = std::chrono::steady_clock::now();
            service_batched_prefill_ms = std::chrono::duration<double, std::milli>(
                batch_end - batch_start).count();
            service_model_batch_calls += batch_prefill_stats.model_batch_calls;
            service_tensor_batch_requests +=
                batch_prefill_stats.tensor_batch_requests;
            service_prefill_group_calls += batch_prefill_stats.prefill_group_calls;
            service_prefill_group_requests +=
                batch_prefill_stats.prefill_group_requests;
            for (std::size_t step = 1;
                 step < 8 &&
                     (!batch_request_a.terminal() || !batch_request_b.terminal());
                 ++step) {
                const auto stats = batch_service.dispatch_step(runtime, 2048);
                service_model_batch_calls += stats.model_batch_calls;
                service_tensor_batch_requests += stats.tensor_batch_requests;
                if (stats.scheduled_requests == 0) break;
            }
            service_batch_workspaces = batch_runner.workspace_count();
            service_packed_prefill_calls = batch_runner.packed_prefill_count();
            service_prefill_workspaces =
                batch_runner.prefill_workspace_count();
            const bool prefill_ab_match = serial_admitted && serial_released &&
                batch_admitted &&
                serial_prefill_stats.scheduled_requests == 2 &&
                serial_prefill_stats.model_batch_calls == 0 &&
                batch_prefill_stats.scheduled_requests == 2 &&
                batch_prefill_stats.prefill_group_calls == 1 &&
                batch_prefill_stats.prefill_group_requests == 2 &&
                serial_request_a.generated() == batch_request_a.generated() &&
                serial_request_b.generated() == batch_request_b.generated();
            service_batch_match = prefill_ab_match &&
                batch_request_a.finished() && batch_request_b.finished() &&
                batch_request_a.generated() == runtime_cuda_tokens &&
                batch_request_b.generated() == runtime_cuda_tokens &&
                service_model_batch_calls == 3 &&
                service_tensor_batch_requests == 6 &&
                service_prefill_group_calls == 1 &&
                service_prefill_group_requests == 2 &&
                service_packed_prefill_calls == 1 &&
                service_prefill_workspaces == 1 &&
                service_batch_workspaces == 1;
            service_batch_released =
                batch_service.release_terminal() == 2 &&
                batch_service.inflight_size() == 0;
            matched = matched && service_sync_match && service_stream_match &&
                service_metrics_match && service_batch_match &&
                service_batch_released;
        }
#else
        if (service_mode) {
            std::cerr << "mixed service requires "
                         "MULTIMODAL_ENABLE_SERVICE_INTEGRATION=ON\n";
            return 1;
        }
#endif
#endif
    }
    std::cout << std::setprecision(10)
              << "decoder_layers: " << cuda_model->layer_count() << '\n'
              << "vision_mode: "
              << (full_tile ? "full-tile" : "top-left-window") << '\n'
              << "prompt_mode: "
              << (mixed_prompt ? "mixed-text-vision" : "visual-only") << '\n'
              << "visual_tokens: " << visual_tokens << '\n'
              << "text_tokens: " << text_tokens << '\n'
              << "prompt_prefix_tokens: " << prompt_prefix_tokens << '\n'
              << "prompt_suffix_tokens: " << prompt_suffix_tokens << '\n'
              << "sequence_tokens: " << embeddings.token_count << '\n'
              << "hidden_size: " << config.projection_dim << '\n'
              << "logits_maximum_error: " << maximum_error << '\n'
              << "logits_cosine_similarity: " << cosine << '\n'
              << "cpu_greedy_token: " << cpu_token << '\n'
              << "cuda_greedy_token: " << cuda_token << '\n'
              << "decode_performed: "
              << (mixed_prompt ? "true" : "false") << '\n'
              << "decode_logits_maximum_error: " << decode_maximum_error << '\n'
              << "decode_logits_cosine_similarity: " << decode_cosine << '\n'
              << "cpu_decode_greedy_token: " << cpu_decode_token << '\n'
              << "cuda_decode_greedy_token: " << cuda_decode_token << '\n'
              << "runtime_generated_tokens: ";
    for (std::size_t index = 0; index < runtime_cuda_tokens.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << runtime_cuda_tokens[index];
    }
    std::cout << '\n'
              << "runtime_generated_text: " << runtime_generated_text << '\n'
              << "runtime_context_positions: " << runtime_cpu_position << ','
              << runtime_cuda_position << '\n'
              << "service_mode: " << (service_mode ? "true" : "false") << '\n'
              << "service_sync_match: "
              << (service_sync_match ? "true" : "false") << '\n'
              << "service_stream_match: "
              << (service_stream_match ? "true" : "false") << '\n'
              << "service_metrics_match: "
              << (service_metrics_match ? "true" : "false") << '\n'
              << "service_factory_calls: " << service_factory_calls << '\n'
              << "service_vision_batch_calls: "
              << service_vision_batch_calls << '\n'
              << "service_vision_max_batch_size: "
              << service_vision_max_batch_size << '\n'
              << "service_async_prefill_group_calls: "
              << service_async_prefill_group_calls << '\n'
              << "service_async_tensor_batch_requests: "
              << service_async_tensor_batch_requests << '\n'
              << "service_async_packed_prefill_calls: "
              << service_async_packed_prefill_calls << '\n'
              << "service_batch_match: "
              << (service_batch_match ? "true" : "false") << '\n'
              << "service_batch_released: "
              << (service_batch_released ? "true" : "false") << '\n'
              << "service_model_batch_calls: "
              << service_model_batch_calls << '\n'
              << "service_tensor_batch_requests: "
              << service_tensor_batch_requests << '\n'
              << "service_prefill_group_calls: "
              << service_prefill_group_calls << '\n'
              << "service_prefill_group_requests: "
              << service_prefill_group_requests << '\n'
              << "service_batch_workspaces: "
              << service_batch_workspaces << '\n'
              << "service_packed_prefill_calls: "
              << service_packed_prefill_calls << '\n'
              << "service_prefill_workspaces: "
              << service_prefill_workspaces << '\n'
              << "service_serial_prefill_ms: "
              << service_serial_prefill_ms << '\n'
              << "service_batched_prefill_ms: "
              << service_batched_prefill_ms << '\n'
              << "context_positions: " << cpu_context->token_count() << ','
              << cuda_context->position() << '\n'
              << "cpu_cuda_match: " << (matched ? "true" : "false") << '\n';
    return matched ? 0 : 1;
}
