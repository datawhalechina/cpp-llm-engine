// Model：比较真实 Qwen1.5-MoE 第一层的逐行与按专家分组执行。
// 计时边界：不含 GGUF 索引和权重加载，只包含 Router、专家与共享专家。
// 通过条件：路由一致、逐元素误差不超过阈值，并输出反向顺序配对样本。
#include "cpp_llm_engine/model/real_moe_feed_forward.h"
#include "gguf_reader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using cpp_llm_engine::model::RealMoeBatchMode;
using cpp_llm_engine::model::RealMoeDispatchStats;
using cpp_llm_engine::model::RealMoeRoute;

struct Measurement {
    double milliseconds = 0.0;
    std::vector<std::vector<float>> output;
    std::vector<RealMoeRoute> routes;
    RealMoeDispatchStats stats;
};

bool run_once(const chapter01::ModelMetadata& metadata,
              const chapter01::RealMoeLayerWeights& weights,
              const std::vector<std::vector<float>>& inputs,
              RealMoeBatchMode mode,
              Measurement* measurement,
              std::string* error) {
    const auto start = Clock::now();
    const bool succeeded =
        cpp_llm_engine::model::run_real_moe_feed_forward_batch(
            metadata, weights, inputs, &measurement->output,
            &measurement->routes, &measurement->stats, error, {}, mode);
    const auto stop = Clock::now();
    measurement->milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();
    return succeeded;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) * 0.5
        : values[middle];
}

bool same_routes(const std::vector<RealMoeRoute>& left,
                 const std::vector<RealMoeRoute>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t token = 0; token < left.size(); ++token) {
        if (left[token].expert_indices != right[token].expert_indices ||
            left[token].weights.size() != right[token].weights.size()) {
            return false;
        }
        for (std::size_t selected = 0;
             selected < left[token].weights.size(); ++selected) {
            if (std::fabs(left[token].weights[selected] -
                          right[token].weights[selected]) > 1.0e-6F) {
                return false;
            }
        }
    }
    return true;
}

float maximum_error(const std::vector<std::vector<float>>& left,
                    const std::vector<std::vector<float>>& right) {
    if (left.size() != right.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float maximum = 0.0F;
    for (std::size_t row = 0; row < left.size(); ++row) {
        if (left[row].size() != right[row].size()) {
            return std::numeric_limits<float>::infinity();
        }
        for (std::size_t column = 0; column < left[row].size(); ++column) {
            maximum = std::max(
                maximum, std::fabs(left[row][column] - right[row][column]));
        }
    }
    return maximum;
}

void print_samples(const char* name, const std::vector<double>& samples) {
    std::cout << name << '_';
    std::cout << "samples_ms:";
    for (const double sample : samples) std::cout << ' ' << sample;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: qwen15_moe_grouped_batch_benchmark <model.gguf> "
                     "[batch=2] [warmup=1] [iterations=5]\n";
        return 2;
    }
    const std::size_t batch = argc >= 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 2;
    const std::size_t warmup = argc >= 4
        ? static_cast<std::size_t>(std::stoul(argv[3])) : 1;
    const std::size_t iterations = argc >= 5
        ? static_cast<std::size_t>(std::stoul(argv[4])) : 5;
    if (batch == 0 || iterations == 0) {
        std::cerr << "batch and iterations must be positive\n";
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

    std::vector<std::vector<float>> inputs(
        batch, std::vector<float>(metadata.hidden_size));
    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t column = 0; column < metadata.hidden_size; ++column) {
            const int pattern = static_cast<int>((column + row * 7) % 29) - 14;
            inputs[row][column] = static_cast<float>(pattern) * 0.0025F;
        }
    }

    for (std::size_t index = 0; index < warmup; ++index) {
        Measurement row_wise;
        Measurement grouped;
        if (!run_once(metadata, weights, inputs, RealMoeBatchMode::RowWise,
                      &row_wise, &error) ||
            !run_once(metadata, weights, inputs, RealMoeBatchMode::Grouped,
                      &grouped, &error)) {
            std::cerr << "MoE batch warmup failed: " << error << '\n';
            return 1;
        }
    }

    std::vector<double> row_wise_samples;
    std::vector<double> grouped_samples;
    float max_error = 0.0F;
    bool routes_match = true;
    RealMoeDispatchStats last_stats;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        Measurement first;
        Measurement second;
        const bool grouped_first = iteration % 2 == 1;
        const auto first_mode = grouped_first
            ? RealMoeBatchMode::Grouped : RealMoeBatchMode::RowWise;
        const auto second_mode = grouped_first
            ? RealMoeBatchMode::RowWise : RealMoeBatchMode::Grouped;
        if (!run_once(metadata, weights, inputs, first_mode, &first, &error) ||
            !run_once(metadata, weights, inputs, second_mode, &second, &error)) {
            std::cerr << "MoE batch measurement failed: " << error << '\n';
            return 1;
        }
        const auto& row_wise = grouped_first ? second : first;
        const auto& grouped = grouped_first ? first : second;
        row_wise_samples.push_back(row_wise.milliseconds);
        grouped_samples.push_back(grouped.milliseconds);
        max_error = std::max(
            max_error, maximum_error(row_wise.output, grouped.output));
        routes_match = routes_match &&
            same_routes(row_wise.routes, grouped.routes);
        last_stats = grouped.stats;
        std::cout << "paired_iteration=" << iteration + 1
                  << " path_order="
                  << (grouped_first ? "grouped-first" : "row-wise-first")
                  << " row_wise_ms=" << row_wise.milliseconds
                  << " grouped_ms=" << grouped.milliseconds << '\n';
    }

    const double row_wise_median = median(row_wise_samples);
    const double grouped_median = median(grouped_samples);
    const bool correct = routes_match && max_error <= 1.0e-4F;
    std::cout << "model_architecture: " << metadata.architecture << '\n'
              << "layer: 0\n"
              << "batch: " << batch << '\n'
              << "warmup: " << warmup << '\n'
              << "iterations: " << iterations << '\n'
              << "active_experts: " << last_stats.active_expert_count << '\n'
              << "assignments: " << last_stats.assignment_count << '\n'
              << "maximum_tokens_per_expert: "
              << last_stats.maximum_tokens_per_expert << '\n'
              << "row_wise_median_ms: " << row_wise_median << '\n'
              << "grouped_median_ms: " << grouped_median << '\n'
              << "grouped_to_row_wise_ratio: "
              << grouped_median / row_wise_median << '\n'
              << "maximum_output_error: " << max_error << '\n'
              << "routes_match: " << (routes_match ? "true" : "false")
              << '\n';
    print_samples("row_wise", row_wise_samples);
    print_samples("grouped", grouped_samples);
    std::cout << "correctness: " << (correct ? "PASS" : "FAIL") << '\n';
    return correct ? 0 : 1;
}
