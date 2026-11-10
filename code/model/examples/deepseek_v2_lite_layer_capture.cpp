// Model：捕获 llama.cpp DeepSeek-V2-Lite 首层中间张量。
// 职责：为教程真实首层提供输入、MLA、FFN 与残差的逐节点参考。
// 运行：deepseek_v2_lite_layer_capture <model.gguf>。
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "cpp_llm_engine/model/real_deepseek2_cpu_model.h"
#include "cpp_llm_engine/model/real_deepseek2_dense_layer.h"
#include "cpp_llm_engine/model/real_moe_feed_forward.h"
#include "gguf_reader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

struct Capture {
    std::map<std::string, std::vector<float>> tensors;
    std::vector<std::string> observed_layer_zero;
    std::string error;
};

struct Comparison {
    double mean_absolute_error = 0.0;
    double cosine_similarity = 0.0;
    float maximum_absolute_error = 0.0F;
};

bool wanted(const std::string& name) {
    static const std::vector<std::string> names{
        "embd", "attn_norm-0", "q-0", "kv_cmpr_pe-0",
        "k_pe-0", "kv_cmpr-0", "kqv_out-0", "ffn_inp-0",
        "ffn_norm-0", "ffn_out-0", "l_out-0", "ffn_inp-1",
        "ffn_norm-1", "ffn_moe_out-1", "ffn_shexp-1", "ffn_out-1",
        "l_out-1", "result_output"};
    return std::find(names.begin(), names.end(), name) != names.end();
}

bool capture_tensor(ggml_tensor* tensor, bool ask, void* user_data) {
    auto* capture = static_cast<Capture*>(user_data);
    const std::string name = tensor->name;
    if (ask) {
        if (name == "embd" ||
            (name.size() > 2 && name.compare(name.size() - 2, 2, "-0") == 0)) {
            if (std::find(capture->observed_layer_zero.begin(),
                          capture->observed_layer_zero.end(), name) ==
                capture->observed_layer_zero.end()) {
                capture->observed_layer_zero.push_back(name);
            }
        }
        return wanted(name);
    }
    if (!wanted(name)) return true;
    if (tensor->type != GGML_TYPE_F32) {
        capture->error = name + " is not Float32";
        return false;
    }
    std::vector<float> values(
        static_cast<std::size_t>(ggml_nelements(tensor)), 0.0F);
    ggml_backend_tensor_get(
        tensor, values.data(), 0, values.size() * sizeof(float));
    capture->tensors[name] = std::move(values);
    return true;
}

void print_summary(const std::string& name,
                   const std::vector<float>& values) {
    double sum = 0.0;
    double square_sum = 0.0;
    float maximum = 0.0F;
    for (const float value : values) {
        sum += value;
        square_sum += static_cast<double>(value) * value;
        maximum = std::max(maximum, std::abs(value));
    }
    const double rms = values.empty()
        ? 0.0
        : std::sqrt(square_sum / static_cast<double>(values.size()));
    std::cout << name << ": elements=" << values.size()
              << " sum=" << sum
              << " rms=" << rms
              << " max_abs=" << maximum << '\n';
}

bool compare(const std::string& name,
             const std::vector<float>& actual,
             const std::vector<float>& expected,
             Comparison* comparison) {
    if (actual.size() != expected.size() || actual.empty()) {
        std::cerr << name << " size mismatch: actual=" << actual.size()
                  << " expected=" << expected.size() << '\n';
        return false;
    }
    double error_sum = 0.0;
    double dot = 0.0;
    double actual_square_sum = 0.0;
    double expected_square_sum = 0.0;
    float maximum = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        error_sum += error;
        maximum = std::max(maximum, error);
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_square_sum += static_cast<double>(actual[i]) * actual[i];
        expected_square_sum += static_cast<double>(expected[i]) * expected[i];
    }
    comparison->mean_absolute_error =
        error_sum / static_cast<double>(actual.size());
    comparison->maximum_absolute_error = maximum;
    const double denominator =
        std::sqrt(actual_square_sum * expected_square_sum);
    comparison->cosine_similarity = denominator == 0.0 ? 1.0 : dot / denominator;
    std::cout << name << "_mae=" << comparison->mean_absolute_error
              << " max_error=" << comparison->maximum_absolute_error
              << " cosine=" << comparison->cosine_similarity << '\n';
    return std::isfinite(comparison->mean_absolute_error) &&
        std::isfinite(comparison->cosine_similarity);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: deepseek_v2_lite_layer_capture <model.gguf>\n";
        return 2;
    }

    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::cerr << "llama.cpp model load failed\n";
        llama_backend_free();
        return 1;
    }

    Capture capture;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 32;
    context_params.n_batch = 2;
    context_params.n_ubatch = 2;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.cb_eval = capture_tensor;
    context_params.cb_eval_user_data = &capture;
    llama_context* context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::cerr << "llama.cpp context creation failed\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_token tokens[] = {0, 1};
    const int status = llama_decode(
        context, llama_batch_get_one(tokens, 2));
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    if (status != 0 || !capture.error.empty()) {
        std::cerr << "DeepSeek2 capture failed: " << capture.error << '\n';
        return 1;
    }

    const std::vector<std::string> required{
        "embd", "attn_norm-0", "q-0", "kv_cmpr_pe-0",
        "k_pe-0", "kv_cmpr-0", "kqv_out-0", "ffn_inp-0",
        "ffn_norm-0", "ffn_out-0", "l_out-0", "ffn_inp-1",
        "ffn_norm-1", "ffn_moe_out-1", "ffn_shexp-1", "ffn_out-1",
        "l_out-1", "result_output"};
    for (const auto& name : required) {
        const auto found = capture.tensors.find(name);
        if (found == capture.tensors.end()) {
            std::cerr << "missing captured tensor: " << name << '\n';
            std::cerr << "observed layer-0 tensors:";
            for (const auto& observed : capture.observed_layer_zero) {
                std::cerr << ' ' << observed;
            }
            std::cerr << '\n';
            return 1;
        }
        print_summary(name, found->second);
    }

    chapter01::GgufModelIndex index;
    chapter01::ModelMetadata metadata;
    chapter01::DeepSeek2DecoderTensorLayout layout;
    chapter01::RealDeepSeek2DenseLayerWeights weights;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error) ||
        !chapter01::ModelMetadataAdapter::from_gguf(
            index.metadata, &metadata, &error) ||
        !chapter01::ModelSchema::validate_deepseek2_gguf(
            metadata, index, &layout, &error) ||
        !chapter01::RealDeepSeek2DenseLayerLoader::load(
            argv[1], index, layout, 0, &weights, &error)) {
        std::cerr << "tutorial layer load failed: " << error << '\n';
        return 1;
    }
    cpp_llm_engine::model::RealDeepSeek2LayerContext layer_context;
    cpp_llm_engine::model::RealDeepSeek2LayerResult combined;
    const auto append = [](std::vector<float>* target,
                           const std::vector<float>& values) {
        target->insert(target->end(), values.begin(), values.end());
    };
    const auto& embeddings = capture.tensors.at("embd");
    if (embeddings.size() != 2 * metadata.hidden_size) {
        std::cerr << "captured embedding shape is not two tokens\n";
        return 1;
    }
    for (std::size_t token = 0; token < 2; ++token) {
        const auto begin = embeddings.begin() + static_cast<std::ptrdiff_t>(
            token * metadata.hidden_size);
        const std::vector<float> input(
            begin, begin + static_cast<std::ptrdiff_t>(metadata.hidden_size));
        cpp_llm_engine::model::RealDeepSeek2LayerResult result;
        if (!cpp_llm_engine::model::run_deepseek2_dense_token_layer(
                metadata, weights, input, &layer_context, &result, &error)) {
            std::cerr << "tutorial layer execution failed: " << error << '\n';
            return 1;
        }
        append(&combined.attention_norm, result.attention_norm);
        append(&combined.query, result.query);
        append(&combined.compressed_kv_with_rope,
               result.compressed_kv_with_rope);
        append(&combined.normalized_latent_kv,
               result.normalized_latent_kv);
        append(&combined.attention_residual, result.attention_residual);
        append(&combined.ffn_norm, result.ffn_norm);
        append(&combined.ffn_output, result.ffn_output);
        append(&combined.output, result.output);
    }

    const std::vector<std::pair<std::string,
        const std::vector<float>*>> comparisons{
        {"attention_norm", &combined.attention_norm},
        {"query", &combined.query},
        {"compressed_kv_with_rope", &combined.compressed_kv_with_rope},
        {"normalized_latent_kv", &combined.normalized_latent_kv},
        {"attention_residual", &combined.attention_residual},
        {"ffn_norm", &combined.ffn_norm},
        {"ffn_output", &combined.ffn_output},
        {"layer_output", &combined.output},
    };
    const std::vector<std::string> expected_names{
        "attn_norm-0", "q-0", "kv_cmpr_pe-0", "kv_cmpr-0",
        "ffn_inp-0", "ffn_norm-0", "ffn_out-0", "l_out-0"};
    double minimum_cosine = 1.0;
    double layer_output_mae = 0.0;
    for (std::size_t i = 0; i < comparisons.size(); ++i) {
        Comparison comparison;
        if (!compare(comparisons[i].first, *comparisons[i].second,
                     capture.tensors.at(expected_names[i]), &comparison)) {
            return 1;
        }
        minimum_cosine = std::min(
            minimum_cosine, comparison.cosine_similarity);
        if (comparisons[i].first == "layer_output") {
            layer_output_mae = comparison.mean_absolute_error;
        }
    }
    if (minimum_cosine < 0.999 || layer_output_mae > 0.002) {
        std::cerr << "tutorial layer comparison exceeded its error budget\n";
        return 1;
    }

    chapter01::RealDeepSeek2MoeLayerWeights moe_weights;
    if (!chapter01::RealDeepSeek2MoeLayerLoader::load(
            argv[1], index, layout, 1, &moe_weights, &error)) {
        std::cerr << "tutorial MoE layer load failed: " << error << '\n';
        return 1;
    }
    const auto& moe_layer_inputs = capture.tensors.at("l_out-0");
    if (moe_layer_inputs.size() != 2 * metadata.hidden_size) {
        std::cerr << "captured MoE layer input shape is not two tokens\n";
        return 1;
    }
    cpp_llm_engine::model::RealDeepSeek2LayerContext moe_context;
    cpp_llm_engine::model::RealDeepSeek2LayerResult combined_moe_layer;
    cpp_llm_engine::model::RealMoeTrace combined_moe_trace;
    for (std::size_t token = 0; token < 2; ++token) {
        const auto begin = moe_layer_inputs.begin() + static_cast<std::ptrdiff_t>(
            token * metadata.hidden_size);
        const std::vector<float> input(
            begin, begin + static_cast<std::ptrdiff_t>(metadata.hidden_size));
        cpp_llm_engine::model::RealDeepSeek2LayerResult layer_result;
        cpp_llm_engine::model::RealMoeRoute route;
        cpp_llm_engine::model::RealMoeTrace trace;
        if (!cpp_llm_engine::model::run_deepseek2_moe_token_layer(
                metadata, moe_weights, input, &moe_context, &layer_result,
                &route, &error, {}, &trace)) {
            std::cerr << "tutorial MoE layer execution failed: "
                      << error << '\n';
            return 1;
        }
        if (route.expert_indices.size() != metadata.expert_used_count ||
            route.shared_expert_weight != 1.0F) {
            std::cerr << "tutorial MoE route does not match DeepSeek2 policy\n";
            return 1;
        }
        append(&combined_moe_layer.attention_residual,
               layer_result.attention_residual);
        append(&combined_moe_layer.ffn_norm, layer_result.ffn_norm);
        append(&combined_moe_layer.ffn_output, layer_result.ffn_output);
        append(&combined_moe_layer.output, layer_result.output);
        append(&combined_moe_trace.routed_output, trace.routed_output);
        append(&combined_moe_trace.weighted_shared_output,
               trace.weighted_shared_output);
    }
    const std::vector<std::pair<std::string,
        const std::vector<float>*>> moe_comparisons{
        {"moe_attention_residual", &combined_moe_layer.attention_residual},
        {"moe_ffn_norm", &combined_moe_layer.ffn_norm},
        {"moe_routed_output", &combined_moe_trace.routed_output},
        {"moe_shared_output", &combined_moe_trace.weighted_shared_output},
        {"moe_ffn_output", &combined_moe_layer.ffn_output},
        {"moe_layer_output", &combined_moe_layer.output},
    };
    const std::vector<std::string> moe_expected_names{
        "ffn_inp-1", "ffn_norm-1", "ffn_moe_out-1", "ffn_shexp-1",
        "ffn_out-1", "l_out-1"};
    double minimum_moe_cosine = 1.0;
    double moe_output_mae = 0.0;
    for (std::size_t i = 0; i < moe_comparisons.size(); ++i) {
        Comparison comparison;
        if (!compare(moe_comparisons[i].first, *moe_comparisons[i].second,
                     capture.tensors.at(moe_expected_names[i]), &comparison)) {
            return 1;
        }
        minimum_moe_cosine = std::min(
            minimum_moe_cosine, comparison.cosine_similarity);
        if (moe_comparisons[i].first == "moe_layer_output") {
            moe_output_mae = comparison.mean_absolute_error;
        }
    }
    if (minimum_moe_cosine < 0.999 || moe_output_mae > 0.005) {
        std::cerr << "tutorial MoE comparison exceeded its error budget\n";
        return 1;
    }

    auto cpu_model =
        cpp_llm_engine::model::RealDeepSeek2CpuModel::load(argv[1], &error);
    if (cpu_model == nullptr) {
        std::cerr << "tutorial DeepSeek2 model load failed: " << error << '\n';
        return 1;
    }
    auto cpu_context = cpu_model->create_context();
    std::vector<float> logits;
    if (!cpu_context->prefill_tokens({0, 1}, &logits, &error)) {
        std::cerr << "tutorial DeepSeek2 model execution failed: "
                  << error << '\n';
        return 1;
    }
    Comparison logits_comparison;
    const auto& reference_logits = capture.tensors.at("result_output");
    if (!compare("model_logits", logits, reference_logits,
                 &logits_comparison)) {
        return 1;
    }
    const auto greedy = [](const std::vector<float>& values) {
        return static_cast<std::size_t>(std::distance(
            values.begin(), std::max_element(values.begin(), values.end())));
    };
    const auto tutorial_token = greedy(logits);
    const auto reference_token = greedy(reference_logits);
    std::cout << "model_layers=" << cpu_context->layer_count() << '\n'
              << "model_position=" << cpu_context->position() << '\n'
              << "tutorial_greedy_token=" << tutorial_token << '\n'
              << "reference_greedy_token=" << reference_token << '\n';
    if (cpu_context->position() != 2 ||
        logits_comparison.cosine_similarity < 0.99 ||
        tutorial_token != reference_token) {
        std::cerr << "tutorial model comparison exceeded its error budget\n";
        return 1;
    }
    std::cout << "deepseek_v2_lite_tutorial_model_compare=PASS\n";
    std::cout << "deepseek_v2_lite_tutorial_moe_compare=PASS\n";
    std::cout << "deepseek_v2_lite_tutorial_layer_compare=PASS\n";
    std::cout << "deepseek_v2_lite_layer_capture=PASS\n";
    return 0;
}
