// Model：验证真实 Qwen1.5-MoE 请求 Context 的 Prefill/Decode 生命周期。
// 运行：qwen15_moe_context_smoke <model.gguf>。
#include "cpp_llm_engine/model/real_qwen2_moe_cpu_model.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::size_t greedy_token(const std::vector<float>& logits) {
    return static_cast<std::size_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: qwen15_moe_context_smoke <model.gguf>\n";
        return 2;
    }
    std::string error;
    auto model = cpp_llm_engine::model::RealQwen2MoeCpuModel::load(
        argv[1], &error);
    if (model == nullptr) {
        std::cerr << "model load failed: " << error << '\n';
        return 1;
    }
    auto context = model->create_context();
    std::vector<float> prefill_logits;
    std::vector<float> decode_logits;
    if (!context->prefill_tokens({0}, &prefill_logits, &error) ||
        context->position() != 1 ||
        !context->decode_token(1, &decode_logits, &error) ||
        context->position() != 2 || prefill_logits.empty() ||
        decode_logits.size() != model->metadata().vocab_size ||
        !finite(prefill_logits) || !finite(decode_logits)) {
        std::cerr << "context execution failed: " << error << '\n';
        return 1;
    }
    const std::size_t position_before_failure = context->position();
    std::vector<float> rejected_logits{7.0F};
    if (context->decode_token(-1, &rejected_logits, &error) ||
        context->position() != position_before_failure ||
        rejected_logits != std::vector<float>{7.0F}) {
        std::cerr << "context failure atomicity failed\n";
        return 1;
    }
    std::cout << "architecture: " << model->metadata().architecture << '\n'
              << "layer_count: " << context->layer_count() << '\n'
              << "committed_tokens: " << context->position() << '\n'
              << "prefill_greedy_token: " << greedy_token(prefill_logits) << '\n'
              << "decode_greedy_token: " << greedy_token(decode_logits) << '\n';
    context->reset();
    if (context->position() != 0) {
        std::cerr << "context reset failed\n";
        return 1;
    }
    std::cout << "qwen15_moe_context: PASS\n";
    return 0;
}
