// Model：用 llama.cpp 中间张量对照 Qwen1.5-MoE 多层前向。
// 职责：捕获 Attention、Top-K 与逐层输出，并比较教程执行器输出。
// 运行：qwen15_moe_ffn_compare <model.gguf> [--tokens 1|2] [--layers N]。
// 通过条件：逐层局部路由、局部数值和累计输出满足各自门槛。
#include "cpp_llm_engine/model/real_moe_feed_forward.h"
#include "cpp_llm_engine/model/real_qwen2_moe_layer.h"
#include "cpp_llm_engine/model/real_qwen2_moe_output.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"
#include "llama.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CapturedFfn {
    std::vector<std::vector<float>> layer_outputs;
    std::vector<std::vector<std::int32_t>> layer_experts;
    std::vector<float> residual_input;
    std::vector<float> ffn_input;
    std::vector<float> input;
    std::vector<float> routed_output;
    std::vector<float> shared_output;
    std::vector<float> output;
    std::vector<float> output_norm;
    std::vector<float> output_logits;
    std::string error;
};

int suffixed_index(const std::string& name, std::string_view prefix) {
    if (name.rfind(prefix, 0) != 0) return -1;
    int index = -1;
    const char* begin = name.data() + prefix.size();
    const char* end = name.data() + name.size();
    const auto parsed = std::from_chars(begin, end, index);
    return parsed.ec == std::errc{} && parsed.ptr == end ? index : -1;
}

bool capture_tensor(ggml_tensor* tensor, bool ask, void* user_data) {
    auto* captured = static_cast<CapturedFfn*>(user_data);
    const std::string name = tensor->name;
    const int output_index = suffixed_index(name, "l_out-");
    const int expert_index = suffixed_index(name, "ffn_moe_topk-");
    const bool wanted = name == "embd" || name == "ffn_inp-0" ||
        name == "ffn_norm-0" || name == "ffn_moe_out-0" ||
        name == "ffn_shexp_out-0" || name == "ffn_out-0" ||
        name == "result_norm" || name == "result_output" ||
        (output_index >= 0 &&
         static_cast<std::size_t>(output_index) <
             captured->layer_outputs.size()) ||
        (expert_index >= 0 &&
         static_cast<std::size_t>(expert_index) <
             captured->layer_experts.size());
    if (ask) return wanted;
    if (!wanted) return true;
    if (expert_index >= 0) {
        if (tensor->type != GGML_TYPE_I32 || tensor->ne[0] <= 0 ||
            tensor->ne[1] <= 0) {
            captured->error = name + " is not Int32";
            return false;
        }
        const std::size_t row_elements =
            static_cast<std::size_t>(tensor->ne[0]);
        const std::size_t rows = static_cast<std::size_t>(tensor->ne[1]);
        const std::size_t elements = row_elements * rows;
        std::vector<std::int32_t> values(elements, 0);
        for (std::size_t row = 0; row < rows; ++row) {
            ggml_backend_tensor_get(
                tensor, values.data() + row * row_elements,
                row * tensor->nb[1], row_elements * sizeof(std::int32_t));
        }
        captured->layer_experts[static_cast<std::size_t>(expert_index)] =
            std::move(values);
        return true;
    }
    if (tensor->type != GGML_TYPE_F32) {
        captured->error = name + " is not Float32";
        return false;
    }
    const std::size_t elements = ggml_nelements(tensor);
    std::vector<float> values(elements, 0.0F);
    ggml_backend_tensor_get(tensor, values.data(), 0,
                            values.size() * sizeof(float));
    if (name == "embd") {
        captured->residual_input = std::move(values);
    } else if (name == "ffn_inp-0") {
        captured->ffn_input = std::move(values);
    } else if (name == "ffn_norm-0") {
        captured->input = std::move(values);
    } else if (name == "ffn_moe_out-0") {
        captured->routed_output = std::move(values);
    } else if (name == "ffn_shexp_out-0") {
        captured->shared_output = std::move(values);
    } else if (name == "ffn_out-0") {
        captured->output = std::move(values);
    } else if (name == "result_norm") {
        captured->output_norm = std::move(values);
    } else if (name == "result_output") {
        captured->output_logits = std::move(values);
    } else if (output_index >= 0) {
        captured->layer_outputs[static_cast<std::size_t>(output_index)] =
            std::move(values);
    }
    return true;
}

struct Metrics {
    float maximum = 0.0F;
    double mean_absolute = 0.0;
    double root_mean_square = 0.0;
    double cosine = 0.0;
};

Metrics compare(const std::vector<float>& actual,
                const std::vector<float>& expected) {
    Metrics result;
    double squared = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double difference =
            static_cast<double>(actual[i]) - expected[i];
        const double absolute = std::fabs(difference);
        result.maximum = std::max(result.maximum,
                                  static_cast<float>(absolute));
        result.mean_absolute += absolute;
        squared += difference * difference;
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
        expected_norm += static_cast<double>(expected[i]) * expected[i];
    }
    result.mean_absolute /= actual.size();
    result.root_mean_square = std::sqrt(squared / actual.size());
    result.cosine = dot / std::sqrt(actual_norm * expected_norm);
    return result;
}

bool parse_positive_int(const std::string& text, int* output) {
    int value = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        value <= 0) {
        return false;
    }
    *output = value;
    return true;
}

std::vector<std::vector<float>> split_tokens(
    const std::vector<float>& values,
    std::size_t token_count,
    std::size_t hidden_size) {
    std::vector<std::vector<float>> tokens;
    tokens.reserve(token_count);
    for (std::size_t token = 0; token < token_count; ++token) {
        const auto begin = values.begin() +
            static_cast<std::ptrdiff_t>(token * hidden_size);
        tokens.emplace_back(begin, begin +
            static_cast<std::ptrdiff_t>(hidden_size));
    }
    return tokens;
}

std::vector<float> flatten_layer_field(
    const std::vector<cpp_llm_engine::model::RealQwen2MoeLayerResult>& tokens,
    const std::vector<float> cpp_llm_engine::model::RealQwen2MoeLayerResult::*field) {
    std::vector<float> values;
    for (const auto& token : tokens) {
        const auto& token_values = token.*field;
        values.insert(values.end(), token_values.begin(), token_values.end());
    }
    return values;
}

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index,
    const std::string& name) {
    const auto found = std::find_if(
        index.tensors.begin(), index.tensors.end(),
        [&name](const chapter01::GgufTensorInfo& tensor) {
            return tensor.name == name;
        });
    return found == index.tensors.end() ? nullptr : &*found;
}

std::vector<std::size_t> top_indices(const std::vector<float>& values,
                                     std::size_t count) {
    std::vector<std::size_t> indices(values.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    const std::size_t selected = std::min(count, indices.size());
    std::partial_sort(
        indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(selected),
        indices.end(), [&values](std::size_t left, std::size_t right) {
            if (values[left] == values[right]) return left < right;
            return values[left] > values[right];
        });
    indices.resize(selected);
    return indices;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6 || argc % 2 != 0) {
        std::cerr << "usage: qwen15_moe_ffn_compare <model.gguf> "
                     "[--tokens 1|2] [--layers N]\n";
        return 2;
    }
    int token_count = 1;
    int layer_count = 1;
    for (int argument = 2; argument < argc; argument += 2) {
        const std::string option = argv[argument];
        const std::string value = argv[argument + 1];
        if (option == "--tokens" && (value == "1" || value == "2")) {
            token_count = std::stoi(value);
        } else if (option == "--layers" &&
                   parse_positive_int(value, &layer_count)) {
        } else {
            std::cerr << "--tokens must be 1 or 2; --layers must be positive\n";
            return 2;
        }
    }

    llama_log_set(
        [](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::cerr << "llama.cpp model load failed\n";
        llama_backend_free();
        return 1;
    }

    CapturedFfn captured;
    captured.layer_outputs.resize(static_cast<std::size_t>(layer_count));
    captured.layer_experts.resize(static_cast<std::size_t>(layer_count));
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 32;
    context_params.n_batch = token_count;
    context_params.n_ubatch = token_count;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.cb_eval = capture_tensor;
    context_params.cb_eval_user_data = &captured;
    llama_context* context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::cerr << "llama.cpp context creation failed\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens(static_cast<std::size_t>(token_count));
    for (int i = 0; i < token_count; ++i) tokens[i] = i;
    const int decode_status =
        llama_decode(context, llama_batch_get_one(tokens.data(), token_count));
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    if (decode_status != 0 || !captured.error.empty() ||
        captured.input.empty() || captured.routed_output.empty() ||
        captured.shared_output.empty() || captured.output.empty() ||
        captured.layer_outputs.empty() || captured.layer_outputs.back().empty()) {
        std::cerr << "llama.cpp FFN capture failed: " << captured.error << '\n';
        return 1;
    }

    chapter01::GgufModelIndex index;
    chapter01::ModelMetadata metadata;
    chapter01::MoeDecoderTensorLayout layout;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error) ||
        !chapter01::ModelMetadataAdapter::from_gguf(
            index.metadata, &metadata, &error) ||
        !chapter01::ModelSchema::validate_moe_gguf(
            metadata, index, &layout, &error) ||
        static_cast<std::size_t>(layer_count) > layout.layers.size()) {
        std::cerr << "tutorial MoE model load failed: " << error << '\n';
        return 1;
    }

    const std::size_t hidden_size = metadata.hidden_size;
    const std::size_t expected_elements =
        static_cast<std::size_t>(token_count) * hidden_size;
    if (captured.residual_input.size() != expected_elements ||
        captured.ffn_input.size() != expected_elements) {
        std::cerr << "layer comparison shape mismatch\n";
        return 1;
    }
    for (std::size_t layer = 0; layer < captured.layer_outputs.size(); ++layer) {
        const bool final_model_layer_uses_last_token =
            layer + 1 == layout.layers.size() &&
            captured.layer_outputs[layer].size() == hidden_size;
        if (captured.layer_outputs[layer].size() != expected_elements &&
            !final_model_layer_uses_last_token) {
            std::cerr << "captured layer " << layer << " has "
                      << captured.layer_outputs[layer].size()
                      << " elements; expected " << expected_elements << '\n';
            return 1;
        }
        const std::size_t expected_experts =
            static_cast<std::size_t>(token_count) *
            metadata.expert_used_count;
        const bool final_model_layer_uses_last_token_route =
            layer + 1 == layout.layers.size() &&
            captured.layer_experts[layer].size() ==
                metadata.expert_used_count;
        if (captured.layer_experts[layer].size() != expected_experts &&
            !final_model_layer_uses_last_token_route) {
            std::cerr << "captured layer " << layer << " has "
                      << captured.layer_experts[layer].size()
                      << " selected experts; expected "
                      << expected_experts
                      << '\n';
            return 1;
        }
    }

    auto layer_inputs = split_tokens(
        captured.residual_input, static_cast<std::size_t>(token_count),
        hidden_size);
    cpp_llm_engine::model::RealQwen2MoePrefillResult first_layer_result;
    chapter01::RealMoeLayerWeights first_layer_weights;
    std::size_t total_key_elements = 0;
    std::size_t total_value_elements = 0;
    std::vector<Metrics> layer_metrics;
    layer_metrics.reserve(static_cast<std::size_t>(layer_count));
    std::vector<Metrics> local_layer_metrics;
    local_layer_metrics.reserve(static_cast<std::size_t>(layer_count));
    std::vector<std::size_t> layer_comparison_elements;
    layer_comparison_elements.reserve(static_cast<std::size_t>(layer_count));
    std::vector<std::size_t> route_mismatch_tokens;
    route_mismatch_tokens.reserve(static_cast<std::size_t>(layer_count));
    std::vector<std::size_t> reference_route_mismatch_tokens;
    reference_route_mismatch_tokens.reserve(
        static_cast<std::size_t>(layer_count));
    std::vector<std::size_t> local_reference_route_mismatch_tokens;
    local_reference_route_mismatch_tokens.reserve(
        static_cast<std::size_t>(layer_count));
    std::vector<std::size_t> route_comparison_tokens;
    route_comparison_tokens.reserve(static_cast<std::size_t>(layer_count));
    std::vector<float> final_local_hidden;
    for (int layer = 0; layer < layer_count; ++layer) {
        chapter01::RealMoeLayerWeights weights;
        if (!chapter01::RealMoeLayerLoader::load(
                argv[1], index, layout, static_cast<std::size_t>(layer),
                &weights, &error)) {
            std::cerr << "tutorial MoE layer load failed: " << error << '\n';
            return 1;
        }
        cpp_llm_engine::model::RealQwen2MoePrefillResult layer_result;
        if (!cpp_llm_engine::model::run_qwen2_moe_prefill_layer(
                metadata, weights, layer_inputs, &layer_result, &error)) {
            std::cerr << "tutorial Qwen2MoE layer failed: " << error << '\n';
            return 1;
        }
        total_key_elements += layer_result.keys.size();
        total_value_elements += layer_result.values.size();
        const auto current_output = flatten_layer_field(
            layer_result.tokens,
            &cpp_llm_engine::model::RealQwen2MoeLayerResult::output);
        const auto& reference_output =
            captured.layer_outputs[static_cast<std::size_t>(layer)];
        if (reference_output.size() == current_output.size()) {
            layer_metrics.push_back(compare(current_output, reference_output));
        } else {
            // llama.cpp only retains the last token at the final model layer
            // because that hidden state is sufficient to produce next-token logits.
            layer_metrics.push_back(compare(
                layer_result.tokens.back().output, reference_output));
        }
        layer_comparison_elements.push_back(reference_output.size());
        const std::size_t captured_route_tokens =
            captured.layer_experts[static_cast<std::size_t>(layer)].size() /
            metadata.expert_used_count;
        const std::size_t route_token_offset =
            static_cast<std::size_t>(token_count) - captured_route_tokens;
        route_comparison_tokens.push_back(captured_route_tokens);
        const auto route_matches_reference =
            [&](const cpp_llm_engine::model::RealQwen2MoeLayerResult& token,
                std::size_t token_index) {
                const auto begin = captured.layer_experts[
                    static_cast<std::size_t>(layer)].begin() +
                    static_cast<std::ptrdiff_t>(
                        (token_index - route_token_offset) *
                        metadata.expert_used_count);
                const auto end = begin + static_cast<std::ptrdiff_t>(
                    metadata.expert_used_count);
                return std::is_permutation(
                    token.route.expert_indices.begin(),
                    token.route.expert_indices.end(), begin, end,
                    [](std::size_t actual, std::int32_t expected) {
                        return actual == static_cast<std::size_t>(expected);
                    });
            };
        std::size_t reference_mismatches = 0;
        for (std::size_t token = route_token_offset;
             token < layer_result.tokens.size();
             ++token) {
            if (!route_matches_reference(layer_result.tokens[token], token)) {
                ++reference_mismatches;
            }
        }
        reference_route_mismatch_tokens.push_back(reference_mismatches);

        if (layer == 0) {
            local_layer_metrics.push_back(layer_metrics.back());
            route_mismatch_tokens.push_back(0);
            local_reference_route_mismatch_tokens.push_back(
                reference_mismatches);
        } else {
            const auto reference_inputs = split_tokens(
                captured.layer_outputs[static_cast<std::size_t>(layer - 1)],
                static_cast<std::size_t>(token_count), hidden_size);
            cpp_llm_engine::model::RealQwen2MoePrefillResult local_result;
            if (!cpp_llm_engine::model::run_qwen2_moe_prefill_layer(
                    metadata, weights, reference_inputs, &local_result,
                    &error)) {
                std::cerr << "tutorial Qwen2MoE local layer failed: "
                          << error << '\n';
                return 1;
            }
            const auto local_output = flatten_layer_field(
                local_result.tokens,
                &cpp_llm_engine::model::RealQwen2MoeLayerResult::output);
            if (reference_output.size() == local_output.size()) {
                local_layer_metrics.push_back(compare(
                    local_output, reference_output));
            } else {
                local_layer_metrics.push_back(compare(
                    local_result.tokens.back().output, reference_output));
            }
            std::size_t mismatches = 0;
            for (std::size_t token = 0; token < layer_result.tokens.size();
                 ++token) {
                if (layer_result.tokens[token].route.expert_indices !=
                    local_result.tokens[token].route.expert_indices) {
                    ++mismatches;
                }
            }
            route_mismatch_tokens.push_back(mismatches);
            std::size_t local_reference_mismatches = 0;
            for (std::size_t token = route_token_offset;
                 token < local_result.tokens.size();
                 ++token) {
                if (!route_matches_reference(local_result.tokens[token], token)) {
                    ++local_reference_mismatches;
                }
            }
            local_reference_route_mismatch_tokens.push_back(
                local_reference_mismatches);
            if (layer + 1 == layer_count) {
                final_local_hidden = local_result.tokens.back().output;
            }
        }
        if (layer == 0) {
            first_layer_result = layer_result;
            first_layer_weights = std::move(weights);
        }
        layer_inputs.clear();
        layer_inputs.reserve(layer_result.tokens.size());
        for (const auto& token : layer_result.tokens) {
            layer_inputs.push_back(token.output);
        }
    }
    const auto attention_output = flatten_layer_field(
        first_layer_result.tokens,
        &cpp_llm_engine::model::RealQwen2MoeLayerResult::attention_residual);
    const Metrics attention_metrics =
        compare(attention_output, captured.ffn_input);
    const Metrics& final_metrics = layer_metrics.back();
    bool passed = std::isfinite(attention_metrics.cosine) &&
        std::all_of(
            layer_metrics.begin(), layer_metrics.end(),
            [](const Metrics& current) {
                return std::isfinite(current.cosine) &&
                    current.mean_absolute <= 0.15 &&
                    current.cosine >= 0.995;
            }) &&
        std::all_of(
            local_layer_metrics.begin(), local_layer_metrics.end(),
            [](const Metrics& current) {
                return std::isfinite(current.cosine) &&
                    current.mean_absolute <= 0.05 &&
                    current.cosine >= 0.9998;
            }) &&
        std::all_of(
            local_reference_route_mismatch_tokens.begin(),
            local_reference_route_mismatch_tokens.end(),
            [](std::size_t mismatches) { return mismatches == 0; }) &&
        attention_metrics.mean_absolute <= 0.02 &&
        attention_metrics.cosine >= 0.99;

    Metrics output_norm_metrics;
    Metrics output_logits_metrics;
    Metrics local_output_norm_metrics;
    Metrics local_output_logits_metrics;
    std::vector<std::size_t> output_top5;
    std::vector<std::size_t> reference_top5;
    std::vector<std::size_t> local_output_top5;
    bool compared_model_output = false;
    if (static_cast<std::size_t>(layer_count) == layout.layers.size()) {
        const auto* output_norm_info = find_tensor(index, layout.output_norm);
        const auto* output_weight_info = find_tensor(index, layout.output);
        chapter01::Tensor output_norm;
        chapter01::Tensor output_weight;
        if (output_norm_info == nullptr || output_weight_info == nullptr ||
            !chapter01::GgufTensorLoader::load_tensor(
                argv[1], index, *output_norm_info, &output_norm, &error) ||
            !chapter01::GgufTensorLoader::load_tensor(
                argv[1], index, *output_weight_info, &output_weight, &error)) {
            std::cerr << "tutorial MoE output weight load failed: "
                      << error << '\n';
            return 1;
        }
        std::vector<float> normalized;
        std::vector<float> logits;
        std::vector<float> local_normalized;
        std::vector<float> local_logits;
        if (final_local_hidden.empty()) final_local_hidden = layer_inputs.back();
        if (!cpp_llm_engine::model::finish_qwen2_moe_output(
                metadata, output_norm, output_weight, layer_inputs.back(),
                &normalized, &logits, &error) ||
            !cpp_llm_engine::model::finish_qwen2_moe_output(
                metadata, output_norm, output_weight, final_local_hidden,
                &local_normalized, &local_logits, &error) ||
            captured.output_norm.size() != normalized.size() ||
            captured.output_logits.size() != logits.size()) {
            std::cerr << "tutorial MoE output failed: " << error << '\n';
            return 1;
        }
        output_norm_metrics = compare(normalized, captured.output_norm);
        output_logits_metrics = compare(logits, captured.output_logits);
        local_output_norm_metrics = compare(
            local_normalized, captured.output_norm);
        local_output_logits_metrics = compare(
            local_logits, captured.output_logits);
        output_top5 = top_indices(logits, 5);
        local_output_top5 = top_indices(local_logits, 5);
        reference_top5 = top_indices(captured.output_logits, 5);
        const auto top5_overlap = [&](const std::vector<std::size_t>& actual) {
            return std::count_if(
                actual.begin(), actual.end(),
                [&](std::size_t token) {
                    return std::find(reference_top5.begin(), reference_top5.end(),
                                     token) != reference_top5.end();
                });
        };
        passed = passed && std::isfinite(output_logits_metrics.cosine) &&
            output_logits_metrics.cosine >= 0.99 &&
            output_top5.front() == reference_top5.front() &&
            top5_overlap(output_top5) >= 4 &&
            std::isfinite(local_output_logits_metrics.cosine) &&
            local_output_logits_metrics.cosine >= 0.999;
        compared_model_output = true;
    }

    Metrics metrics;
    Metrics routed_metrics;
    Metrics shared_metrics;
    cpp_llm_engine::model::RealMoeRoute route;
    if (token_count == 1) {
        std::vector<float> actual;
        cpp_llm_engine::model::RealMoeTrace trace;
        if (!cpp_llm_engine::model::run_real_moe_feed_forward(
                metadata, first_layer_weights, captured.input, &actual, &route,
                &error, {},
                &trace) || actual.size() != captured.output.size()) {
            std::cerr << "tutorial MoE FFN failed: " << error << '\n';
            return 1;
        }
        metrics = compare(actual, captured.output);
        routed_metrics = compare(trace.routed_output, captured.routed_output);
        shared_metrics = compare(
            trace.weighted_shared_output, captured.shared_output);
        passed = passed && std::isfinite(metrics.cosine) &&
            metrics.mean_absolute <= 0.02 &&
            metrics.root_mean_square <= 0.03 && metrics.cosine >= 0.99;
    }
    std::cout << "reference: llama.cpp cb_eval\n"
              << "token count: " << token_count << '\n'
              << "layer count: " << layer_count << '\n'
              << "hidden size: " << hidden_size << '\n'
              << "captured residual size: " << captured.residual_input.size() << '\n'
              << "captured ffn input size: " << captured.ffn_input.size() << '\n'
              << "captured layer output size: "
              << captured.layer_outputs.back().size() << '\n'
              << "key payload size: " << total_key_elements << '\n'
              << "value payload size: " << total_value_elements << '\n';
    if (token_count == 1) {
        std::cout << "selected experts:";
        for (const auto expert : route.expert_indices) std::cout << ' ' << expert;
        std::cout << "\nrouted mae: " << routed_metrics.mean_absolute << '\n'
                  << "routed cosine: " << routed_metrics.cosine << '\n'
                  << "shared mae: " << shared_metrics.mean_absolute << '\n'
                  << "shared cosine: " << shared_metrics.cosine << '\n';
    }
    std::cout << "attention residual mae: "
              << attention_metrics.mean_absolute << '\n'
              << "attention residual cosine: " << attention_metrics.cosine << '\n'
              << "layer output mae: " << final_metrics.mean_absolute << '\n'
              << "layer output cosine: " << final_metrics.cosine << '\n';
    for (std::size_t layer = 0; layer < layer_metrics.size(); ++layer) {
        std::cout << "layer " << layer << " comparison elements: "
                  << layer_comparison_elements[layer] << '\n'
                  << "layer " << layer << " mae: "
                  << layer_metrics[layer].mean_absolute << '\n'
                  << "layer " << layer << " cosine: "
                  << layer_metrics[layer].cosine << '\n'
                  << "layer " << layer << " local mae: "
                  << local_layer_metrics[layer].mean_absolute << '\n'
                  << "layer " << layer << " local cosine: "
                  << local_layer_metrics[layer].cosine << '\n'
                  << "layer " << layer << " route mismatch tokens: "
                  << route_mismatch_tokens[layer] << '\n'
                  << "layer " << layer << " reference route mismatch tokens: "
                  << reference_route_mismatch_tokens[layer] << '\n'
                  << "layer " << layer
                  << " local reference route mismatch tokens: "
                  << local_reference_route_mismatch_tokens[layer] << '\n'
                  << "layer " << layer << " route comparison tokens: "
                  << route_comparison_tokens[layer] << '\n';
    }
    if (token_count == 1) {
        std::cout << "max error: " << metrics.maximum << '\n'
                  << "mean absolute error: " << metrics.mean_absolute << '\n'
                  << "rmse: " << metrics.root_mean_square << '\n'
                  << "cosine similarity: " << metrics.cosine << '\n';
    }
    const auto maximum_local_mae = std::max_element(
        local_layer_metrics.begin(), local_layer_metrics.end(),
        [](const Metrics& left, const Metrics& right) {
            return left.mean_absolute < right.mean_absolute;
        });
    const auto minimum_local_cosine = std::min_element(
        local_layer_metrics.begin(), local_layer_metrics.end(),
        [](const Metrics& left, const Metrics& right) {
            return left.cosine < right.cosine;
        });
    std::size_t cumulative_route_mismatches = 0;
    for (const auto mismatches : reference_route_mismatch_tokens) {
        cumulative_route_mismatches += mismatches;
    }
    std::size_t local_route_mismatches = 0;
    for (const auto mismatches : local_reference_route_mismatch_tokens) {
        local_route_mismatches += mismatches;
    }
    std::cout << "maximum local layer mae: "
              << maximum_local_mae->mean_absolute << '\n'
              << "minimum local layer cosine: "
              << minimum_local_cosine->cosine << '\n'
              << "cumulative reference route mismatch tokens: "
              << cumulative_route_mismatches << '\n'
              << "local reference route mismatch tokens: "
              << local_route_mismatches << '\n';
    if (compared_model_output) {
        std::cout << "output norm mae: "
                  << output_norm_metrics.mean_absolute << '\n'
                  << "output norm cosine: " << output_norm_metrics.cosine << '\n'
                  << "output logits mae: "
                  << output_logits_metrics.mean_absolute << '\n'
                  << "output logits cosine: "
                  << output_logits_metrics.cosine << '\n'
                  << "local output norm mae: "
                  << local_output_norm_metrics.mean_absolute << '\n'
                  << "local output norm cosine: "
                  << local_output_norm_metrics.cosine << '\n'
                  << "local output logits mae: "
                  << local_output_logits_metrics.mean_absolute << '\n'
                  << "local output logits cosine: "
                  << local_output_logits_metrics.cosine << '\n'
                  << "greedy token: " << output_top5.front() << '\n'
                  << "local greedy token: " << local_output_top5.front() << '\n'
                  << "reference greedy token: " << reference_top5.front() << '\n'
                  << "output top5:";
        for (const auto token : output_top5) std::cout << ' ' << token;
        std::cout << "\nlocal output top5:";
        for (const auto token : local_output_top5) std::cout << ' ' << token;
        std::cout << "\nreference top5:";
        for (const auto token : reference_top5) std::cout << ' ' << token;
        std::cout << '\n';
    }
    std::cout << "numeric comparison: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
