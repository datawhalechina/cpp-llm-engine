// Chapter 01/06：真实 GGUF Q6_K 解码对照工具。
// 职责：加载真实 Q6_K tensor，比较教程实现与 llama.cpp 官方 reference。
// 验证：输出元素数量、最大绝对误差和最大相对误差，确认 block 解码一致。
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"

#include "chapter05_quantization/q6k_dequantize.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool valid_shape(const chapter01::GgufTensorInfo& info) {
    return info.dimensions.size() == 2 && info.dimensions[0] > 0 && info.dimensions[1] > 0 &&
           info.dimensions[0] <= std::numeric_limits<std::size_t>::max() &&
           info.dimensions[1] <= std::numeric_limits<std::size_t>::max() &&
           info.dimensions[1] % chapter05_quantization::kQ6KBlockElements == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: gguf_q6k_compare <model.gguf>\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error)) {
        std::cerr << "index load failed: " << error << '\n';
        return 1;
    }

    const chapter01::GgufTensorInfo* selected = nullptr;
    for (const auto& info : index.tensors) {
        if (info.type == 14 && valid_shape(info)) {
            selected = &info;
            if (info.name == "blk.0.ffn_down.weight") break;
        }
    }
    if (selected == nullptr) {
        std::cerr << "no suitable Q6_K 2D tensor found\n";
        return 1;
    }

    chapter01::Tensor tensor;
    if (!chapter01::GgufTensorLoader::load_tensor(argv[1], index, *selected, &tensor, &error)) {
        std::cerr << "tensor load failed: " << error << '\n';
        return 1;
    }

    const std::size_t element_count = tensor.numel();
    std::vector<float> tutorial_output;
    if (!chapter05_quantization::dequantize_q6_k(
            reinterpret_cast<const std::uint8_t*>(tensor.raw_data()), tensor.bytes(),
            element_count, &tutorial_output, &error)) {
        std::cerr << "tutorial Q6_K decode failed: " << error << '\n';
        return 1;
    }

    std::vector<float> llama_output(element_count, 0.0f);
    const auto* blocks = reinterpret_cast<const block_q6_K*>(tensor.raw_data());
    dequantize_row_q6_K(blocks, llama_output.data(), static_cast<int64_t>(element_count));

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (std::size_t i = 0; i < element_count; ++i) {
        const float abs_error = std::abs(tutorial_output[i] - llama_output[i]);
        const float denominator = std::max(std::abs(llama_output[i]), 1e-12f);
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, abs_error / denominator);
    }

    std::cout << "tensor: " << tensor.name() << '\n'
              << "shape: [" << tensor.shape()[0] << ", " << tensor.shape()[1] << "]\n"
              << "elements: " << element_count << '\n'
              << "payload_bytes: " << tensor.bytes() << '\n'
              << "max_abs_error: " << max_abs_error << '\n'
              << "max_rel_error: " << max_rel_error << '\n';
    return max_abs_error <= 1e-6f ? 0 : 1;
}
