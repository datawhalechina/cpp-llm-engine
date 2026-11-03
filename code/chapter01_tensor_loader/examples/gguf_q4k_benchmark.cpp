// Chapter 01/06：真实 GGUF Q4_K benchmark example
// 职责：从 GGUF index 读取真实 Q4_K tensor，并调用 Chapter 05 Linear 路径。
// 输入：model.gguf；输出：reference/fused 多 tensor 延迟和误差。
// 验证：真实 TinyLlama benchmark 的 max_abs_delta 小于约定阈值。
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"

#include "chapter05_quantization/q4k_linear.h"
#include "chapter05_quantization/q4k_dequantize.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t warmup = 1;
    std::size_t iterations = 3;
    std::size_t max_tensors = 3;
};

struct Statistics {
    double min = 0.0;
    double mean = 0.0;
    double p50 = 0.0;
    double max = 0.0;
};

bool parse_size(const char* value, std::size_t* result) {
    try {
        const std::string text(value);
        std::size_t consumed = 0;
        const auto parsed = std::stoull(text, &consumed);
        if (consumed != text.size() || parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        *result = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_options(int argc, char** argv, Options* options) {
    for (int i = 2; i < argc; ++i) {
        if (i + 1 >= argc) return false;
        std::size_t value = 0;
        if (!parse_size(argv[i + 1], &value)) return false;
        const std::string key(argv[i]);
        if (key == "--warmup") {
            options->warmup = value;
        } else if (key == "--iterations") {
            options->iterations = value;
        } else if (key == "--max-tensors") {
            options->max_tensors = value;
        } else {
            return false;
        }
        ++i;
    }
    return true;
}

Statistics summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    Statistics result;
    result.min = samples.front();
    result.max = samples.back();
    result.mean = 0.0;
    for (const double sample : samples) result.mean += sample;
    result.mean /= static_cast<double>(samples.size());
    result.p50 = samples[(samples.size() - 1) / 2];
    return result;
}

bool run_linear(bool fused,
                const chapter01::Tensor& tensor,
                const std::vector<float>& input,
                std::vector<float>* output,
                std::string* error) {
    const auto& shape = tensor.shape();
    const std::size_t rows = static_cast<std::size_t>(shape[0]);
    const std::size_t cols = static_cast<std::size_t>(shape[1]);
    const auto* weights = reinterpret_cast<const std::uint8_t*>(tensor.raw_data());
    if (fused) {
        return chapter05_quantization::Q4KLinearFused::forward(
            weights, tensor.bytes(), rows, cols, input, output, error);
    }
    return chapter05_quantization::Q4KLinearReference::forward(
        weights, tensor.bytes(), rows, cols, input, output, error);
}

std::vector<double> measure(bool fused,
                            const chapter01::Tensor& tensor,
                            const std::vector<float>& input,
                            const Options& options,
                            std::string* error) {
    std::vector<float> output;
    for (std::size_t i = 0; i < options.warmup; ++i) {
        if (!run_linear(fused, tensor, input, &output, error)) return {};
    }
    std::vector<double> samples;
    samples.reserve(options.iterations);
    for (std::size_t i = 0; i < options.iterations; ++i) {
        const auto start = Clock::now();
        if (!run_linear(fused, tensor, input, &output, error)) return {};
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return samples;
}

bool valid_shape(const chapter01::GgufTensorInfo& info) {
    return info.dimensions.size() == 2 && info.dimensions[0] > 0 && info.dimensions[1] > 0 &&
           info.dimensions[0] <= std::numeric_limits<std::size_t>::max() &&
           info.dimensions[1] <= std::numeric_limits<std::size_t>::max() &&
           info.dimensions[1] % chapter05_quantization::kQ4KBlockElements == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: gguf_q4k_benchmark <model.gguf> "
                     "[--warmup N] [--iterations N] [--max-tensors N]\n";
        return 2;
    }
    Options options;
    if (!parse_options(argc, argv, &options)) {
        std::cerr << "invalid benchmark options\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error)) {
        std::cerr << "index load failed: " << error << '\n';
        return 1;
    }

    std::vector<const chapter01::GgufTensorInfo*> candidates;
    for (const auto& info : index.tensors) {
        if (info.type == 12 && valid_shape(info)) candidates.push_back(&info);
    }
    if (candidates.empty()) {
        std::cerr << "no suitable Q4_K 2D tensors found\n";
        return 1;
    }
    if (candidates.size() > options.max_tensors) candidates.resize(options.max_tensors);

    std::cout << "warmup: " << options.warmup << '\n'
              << "iterations: " << options.iterations << '\n'
              << "tensors: " << candidates.size() << '\n';

    for (const auto* info : candidates) {
        chapter01::Tensor tensor;
        if (!chapter01::GgufTensorLoader::load_tensor(argv[1], index, *info, &tensor, &error)) {
            std::cerr << "tensor load failed for " << info->name << ": " << error << '\n';
            return 1;
        }
        const auto& shape = tensor.shape();
        const std::size_t rows = static_cast<std::size_t>(shape[0]);
        const std::size_t cols = static_cast<std::size_t>(shape[1]);
        const std::vector<float> input(cols, 1.0f);

        std::vector<float> reference_output;
        std::vector<float> fused_output;
        if (!run_linear(false, tensor, input, &reference_output, &error) ||
            !run_linear(true, tensor, input, &fused_output, &error)) {
            std::cerr << "correctness run failed for " << info->name << ": " << error << '\n';
            return 1;
        }
        if (reference_output.size() != fused_output.size()) {
            std::cerr << "output size mismatch for " << info->name << '\n';
            return 1;
        }
        float max_abs_delta = 0.0f;
        for (std::size_t i = 0; i < reference_output.size(); ++i) {
            max_abs_delta = std::max(max_abs_delta, std::abs(reference_output[i] - fused_output[i]));
        }

        const auto reference_samples = measure(false, tensor, input, options, &error);
        const auto fused_samples = measure(true, tensor, input, options, &error);
        if (reference_samples.empty() || fused_samples.empty()) {
            std::cerr << "timed run failed for " << info->name << ": " << error << '\n';
            return 1;
        }
        const auto reference = summarize(reference_samples);
        const auto fused = summarize(fused_samples);
        std::cout << "tensor: " << tensor.name() << '\n'
                  << "shape: [" << rows << ", " << cols << "]\n"
                  << "payload_bytes: " << tensor.bytes() << '\n'
                  << "reference_ms: min=" << reference.min << " mean=" << reference.mean
                  << " p50=" << reference.p50 << " max=" << reference.max << '\n'
                  << "fused_ms: min=" << fused.min << " mean=" << fused.mean
                  << " p50=" << fused.p50 << " max=" << fused.max << '\n'
                  << "max_abs_delta: " << max_abs_delta << '\n';
        if (max_abs_delta > 1e-3f) {
            std::cerr << "correctness threshold exceeded for " << info->name << '\n';
            return 1;
        }
    }
    return 0;
}
