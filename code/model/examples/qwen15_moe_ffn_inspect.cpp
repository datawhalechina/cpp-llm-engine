// Model：执行真实 Qwen1.5-MoE layer-0 的量化 FFN 子层。
// 职责：展示 60 选 4 路由、packed expert 切片和共享专家合并。
// 运行：qwen15_moe_ffn_inspect <model.gguf>。
// 通过条件：输出四个专家、归一化权重和有限的输出摘要。
#include "cpp_llm_engine/model/real_moe_feed_forward.h"
#include "gguf_reader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: qwen15_moe_ffn_inspect <model.gguf>\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    chapter01::ModelMetadata metadata;
    chapter01::MoeDecoderTensorLayout layout;
    chapter01::RealMoeLayerWeights weights;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error) ||
        !chapter01::ModelMetadataAdapter::from_gguf(
            index.metadata, &metadata, &error) ||
        !chapter01::ModelSchema::validate_moe_gguf(
            metadata, index, &layout, &error) ||
        !chapter01::RealMoeLayerLoader::load(
            argv[1], index, layout, 0, &weights, &error)) {
        std::cerr << "Qwen1.5-MoE layer load failed: " << error << '\n';
        return 1;
    }

    std::vector<float> hidden(metadata.hidden_size, 0.0F);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01F;
    }
    std::vector<float> output;
    cpp_llm_engine::model::RealMoeRoute route;
    if (!cpp_llm_engine::model::run_real_moe_feed_forward(
            metadata, weights, hidden, &output, &route, &error)) {
        std::cerr << "Qwen1.5-MoE FFN failed: " << error << '\n';
        return 1;
    }

    const bool finite = std::all_of(
        output.begin(), output.end(), [](float value) { return std::isfinite(value); });
    const float route_sum =
        std::accumulate(route.weights.begin(), route.weights.end(), 0.0F);
    float output_l1 = 0.0F;
    for (const float value : output) output_l1 += std::fabs(value);

    std::cout << "architecture: " << metadata.architecture << '\n'
              << "layer: 0\n"
              << "experts selected/total: " << route.expert_indices.size()
              << '/' << metadata.expert_count << '\n'
              << "selected experts:";
    for (const auto expert : route.expert_indices) std::cout << ' ' << expert;
    std::cout << "\nroute weights:";
    for (const auto weight : route.weights) std::cout << ' ' << weight;
    std::cout << "\nroute weight sum: " << route_sum << '\n'
              << "shared expert weight: " << route.shared_expert_weight << '\n'
              << "output size: " << output.size() << '\n'
              << "output l1: " << output_l1 << '\n'
              << "output finite: " << (finite ? "true" : "false") << '\n';
    return finite && output.size() == metadata.hidden_size &&
                   route.expert_indices.size() == metadata.expert_used_count &&
                   route_sum > 0.0F && route_sum <= 1.0F
        ? 0
        : 1;
}
