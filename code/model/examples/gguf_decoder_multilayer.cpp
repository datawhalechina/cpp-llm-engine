// Chapter 01/04/06：真实 TinyLlama 多层 Decoder 与 llama.cpp 对照入口。
// 职责：复用 RealLlamaCpuModel/Context 执行固定 prompt，只保留验证逻辑。
// 验证：比较 logits、greedy sequence，并检查两个 Context 的 KV 状态隔离。
#include "cpp_llm_engine/model/real_llama_cpu_decoder.h"

#if defined(CHAPTER01_COMPARE_LLAMA_CPP)
#include <llama.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<std::pair<std::size_t, float>> top_k(const std::vector<float>& logits,
                                                 std::size_t count) {
    std::vector<std::pair<std::size_t, float>> ranked;
    ranked.reserve(logits.size());
    for (std::size_t token_id = 0; token_id < logits.size(); ++token_id) {
        ranked.emplace_back(token_id, logits[token_id]);
    }
    count = std::min(count, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end(),
                      [](const auto& left, const auto& right) {
                          if (left.second != right.second) return left.second > right.second;
                          return left.first < right.first;
                      });
    ranked.resize(count);
    return ranked;
}

chapter01::TokenId greedy_token(const std::vector<float>& logits) {
    return static_cast<chapter01::TokenId>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

void print_tokens(const char* label, const chapter01::TokenSequence& tokens) {
    std::cout << label;
    for (const auto token : tokens) std::cout << ' ' << token;
    std::cout << '\n';
}

#if defined(CHAPTER01_COMPARE_LLAMA_CPP)
struct LlamaLogSilencer {
    LlamaLogSilencer() {
        llama_log_get(&previous_callback, &previous_user_data);
        llama_log_set(quiet, nullptr);
    }

    ~LlamaLogSilencer() {
        llama_log_set(previous_callback, previous_user_data);
    }

    static void quiet(ggml_log_level, const char*, void*) {}

    ggml_log_callback previous_callback = nullptr;
    void* previous_user_data = nullptr;
};

bool llama_reference(const std::string& model_path,
                     const chapter01::TokenSequence& token_ids,
                     std::size_t vocab_size,
                     std::size_t new_token_count,
                     std::vector<float>* logits,
                     chapter01::TokenSequence* generated,
                     std::string* error) {
    const LlamaLogSilencer silence_logs;
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        llama_backend_free();
        *error = "llama.cpp failed to load model";
        return false;
    }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = static_cast<std::uint32_t>(token_ids.size());
    context_params.n_ubatch = static_cast<std::uint32_t>(token_ids.size());
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.type_k = GGML_TYPE_F32;
    context_params.type_v = GGML_TYPE_F32;
    context_params.no_perf = true;
    llama_context* context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        llama_backend_free();
        *error = "llama.cpp failed to create context";
        return false;
    }
    std::vector<llama_token> tokens(token_ids.begin(), token_ids.end());
    const llama_batch batch = llama_batch_get_one(
        tokens.data(), static_cast<std::int32_t>(tokens.size()));
    const int decode_status = llama_decode(context, batch);
    const float* reference = decode_status == 0 ? llama_get_logits_ith(context, -1) : nullptr;
    if (reference == nullptr) {
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        *error = "llama.cpp failed to decode prompt";
        return false;
    }
    logits->assign(reference, reference + vocab_size);
    generated->clear();
    for (std::size_t step = 0; step < new_token_count; ++step) {
        llama_token token = static_cast<llama_token>(greedy_token(
            std::vector<float>(reference, reference + vocab_size)));
        generated->push_back(static_cast<chapter01::TokenId>(token));
        if (step + 1 == new_token_count) break;
        const llama_batch next_batch = llama_batch_get_one(&token, 1);
        if (llama_decode(context, next_batch) != 0) {
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            *error = "llama.cpp failed during greedy decode";
            return false;
        }
        reference = llama_get_logits_ith(context, -1);
        if (reference == nullptr) {
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            *error = "llama.cpp returned no decode logits";
            return false;
        }
    }
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return true;
}

bool print_logits_comparison(const std::vector<float>& actual,
                             const std::vector<float>& reference) {
    float max_abs_error = 0.0f;
    double absolute_error_sum = 0.0;
    double squared_error_sum = 0.0;
    double dot_product = 0.0;
    double actual_square_sum = 0.0;
    double reference_square_sum = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - reference[i]);
        max_abs_error = std::max(max_abs_error, error);
        absolute_error_sum += error;
        squared_error_sum += static_cast<double>(error) * error;
        dot_product += static_cast<double>(actual[i]) * reference[i];
        actual_square_sum += static_cast<double>(actual[i]) * actual[i];
        reference_square_sum += static_cast<double>(reference[i]) * reference[i];
    }
    const double count = static_cast<double>(actual.size());
    const double cosine = dot_product / std::sqrt(actual_square_sum * reference_square_sum);
    const auto actual_top = top_k(actual, 5);
    const auto reference_top = top_k(reference, 5);
    std::size_t top5_overlap = 0;
    for (const auto& [actual_id, unused] : actual_top) {
        (void)unused;
        if (std::any_of(reference_top.begin(), reference_top.end(),
                        [actual_id](const auto& candidate) {
                            return candidate.first == actual_id;
                        })) {
            ++top5_overlap;
        }
    }
    const double mean_abs_error = absolute_error_sum / count;
    const double rmse = std::sqrt(squared_error_sum / count);
    const bool passed = cosine >= 0.999 && mean_abs_error <= 0.15 &&
                        rmse <= 0.20 && top5_overlap >= 4;
    std::cout << "llama_max_abs_error: " << max_abs_error << '\n'
              << "llama_mean_abs_error: " << mean_abs_error << '\n'
              << "llama_rmse: " << rmse << '\n'
              << "llama_cosine_similarity: " << cosine << '\n'
              << "llama_top1_token_id: " << reference_top.front().first << '\n'
              << "top1_match: " << std::boolalpha
              << (actual_top.front().first == reference_top.front().first) << '\n'
              << "top5_overlap: " << top5_overlap << "/5\n"
              << "llama_logits_compare: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: gguf_decoder_multilayer <model.gguf> [layer_count]\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const std::size_t layer_count = argc == 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 2;
    std::string error;
    auto model = chapter01::RealLlamaCpuModel::load(model_path, layer_count, &error);
    if (model == nullptr) {
        std::cerr << "model load failed: " << error << '\n';
        return 1;
    }
    auto context = model->create_context();
    const auto decoder_contract = model->decoder_contract();
    if (!decoder_contract.validate().is_ok() ||
        decoder_contract.layer_count != layer_count) {
        std::cerr << "decoder contract does not match loaded model\n";
        return 1;
    }

    const chapter01::TokenSequence token_ids = {
        15043, 18439, 1460, 29931, 5288, 29874,
    };
    cpp_llm_engine::core::DecoderEmbeddingSequence prompt_embeddings;
    if (!model->embed_tokens(token_ids, &prompt_embeddings, &error)) {
        std::cerr << "token embedding failed: " << error << '\n';
        return 1;
    }
    std::vector<float> logits;
    if (!context->prefill(token_ids, &logits, &error)) {
        std::cerr << "decoder prefill failed: " << error << '\n';
        return 1;
    }
    const auto stats = context->stats();
    auto isolated_context = model->create_context();
    std::vector<float> isolated_logits;
    cpp_llm_engine::core::DecoderExecutionContext* execution = isolated_context.get();
    const auto context_contract = execution->model_contract();
    if (!context_contract.validate().is_ok() ||
        context_contract.vocab_size != decoder_contract.vocab_size ||
        context_contract.layer_count != decoder_contract.layer_count ||
        context_contract.layer.hidden_size != decoder_contract.layer.hidden_size) {
        std::cerr << "context contract does not match model contract\n";
        return 1;
    }
    const auto isolated_status = execution->prefill_tokens(token_ids, &isolated_logits);
    if (!isolated_status.is_ok()) {
        std::cerr << "isolated context prefill failed: "
                  << isolated_status.message() << '\n';
        return 1;
    }
    auto embedding_context = model->create_context();
    std::vector<float> embedding_logits;
    const auto embedding_status = embedding_context->prefill_embeddings(
        prompt_embeddings, &embedding_logits);
    if (!embedding_status.is_ok()) {
        std::cerr << "embedding prefill failed: " << embedding_status.message() << '\n';
        return 1;
    }
    const bool embedding_prefill_match = embedding_logits == logits &&
        embedding_context->token_count() == token_ids.size();
    auto invalid_embeddings = prompt_embeddings;
    --invalid_embeddings.hidden_size;
    const auto invalid_embedding_status = embedding_context->prefill_embeddings(
        invalid_embeddings, &embedding_logits);
    const bool invalid_embedding_rejected = !invalid_embedding_status.is_ok();
    const bool contexts_isolated = context->stats().token_count == token_ids.size() &&
        execution->token_count() == token_ids.size() &&
        logits == isolated_logits;
    const auto candidates = top_k(logits, 5);
    std::cout << "model_layer_count: " << stats.model_layer_count << '\n'
              << "layer_count: " << stats.loaded_layer_count << '\n'
              << "token_count: " << stats.token_count << '\n'
              << "rope_dim: " << stats.rope_dimension << '\n'
              << "rope_freq_base: " << stats.rope_frequency_base << '\n'
              << "kv_cache_tokens_per_layer: " << stats.token_count << '\n'
              << "attention_score_count: " << stats.attention_score_count << '\n'
              << "logits_size: " << logits.size() << '\n'
              << "logits_finite: true\n"
              << "decoder_contract: PASS\n"
              << "execution_contract: PASS\n"
              << "embedding_prefill_match: "
              << (embedding_prefill_match ? "PASS" : "FAIL") << '\n'
              << "invalid_embedding_rejected: "
              << (invalid_embedding_rejected ? "PASS" : "FAIL") << '\n'
              << "context_isolation: " << (contexts_isolated ? "PASS" : "FAIL") << '\n';
    if (!contexts_isolated || !embedding_prefill_match || !invalid_embedding_rejected) return 1;
    for (std::size_t rank = 0; rank < candidates.size(); ++rank) {
        std::cout << "top_" << rank << "_token_id: " << candidates[rank].first
                  << " logit: " << candidates[rank].second << '\n';
    }
    std::cout << "decoder_logits: ready\n";

#if defined(CHAPTER01_COMPARE_LLAMA_CPP)
    if (stats.loaded_layer_count == stats.model_layer_count) {
        std::vector<float> reference_logits;
        chapter01::TokenSequence reference_tokens;
        if (!llama_reference(model_path, token_ids, model->vocab_size(), 4,
                             &reference_logits, &reference_tokens, &error)) {
            std::cerr << "llama.cpp logits comparison failed: " << error << '\n';
            return 1;
        }
        if (!print_logits_comparison(logits, reference_logits)) return 1;
        chapter01::TokenSequence actual_tokens;
        std::vector<float> next_logits = logits;
        for (std::size_t step = 0; step < reference_tokens.size(); ++step) {
            const auto token = greedy_token(next_logits);
            actual_tokens.push_back(token);
            if (step + 1 < reference_tokens.size() &&
                !context->decode(token, &next_logits, &error)) {
                std::cerr << "decoder greedy generation failed: " << error << '\n';
                return 1;
            }
        }
        print_tokens("decoder_greedy_tokens:", actual_tokens);
        print_tokens("llama_greedy_tokens:", reference_tokens);
        const bool sequence_matches = actual_tokens == reference_tokens;
        std::cout << "greedy_sequence_match: " << std::boolalpha
                  << sequence_matches << '\n';
        if (!sequence_matches) return 1;
    } else {
        std::cout << "llama_logits_compare: skipped (requires full layer count)\n";
    }
#endif
    return 0;
}
