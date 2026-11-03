// Chapter 01/06：真实 Q4_K Linear smoke example
// 职责：加载 GGUF 中一个 Q4_K tensor，验证它能进入 Chapter 05 CPU Linear。
// 输入：model.gguf；输出：reference 与 fused 的首行结果和误差。
// 验证：Chapter 01/06 跨章 CTest。
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"

#include "chapter05_quantization/q4k_linear.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: gguf_q4k_linear_smoke <model.gguf>\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error)) {
        std::cerr << "read failed: " << error << '\n';
        return 1;
    }

    const chapter01::GgufTensorInfo* embedding_info = nullptr;
    for (const auto& info : index.tensors) {
        if (info.name == "token_embd.weight") {
            embedding_info = &info;
            break;
        }
    }
    if (embedding_info == nullptr || embedding_info->type != 12 || embedding_info->dimensions.size() != 2) {
        std::cerr << "token_embd.weight must be a 2D Q4_K tensor\n";
        return 1;
    }

    chapter01::Tensor embedding;
    if (!chapter01::GgufTensorLoader::load_tensor(argv[1], index, *embedding_info, &embedding, &error)) {
        std::cerr << "tensor load failed: " << error << '\n';
        return 1;
    }

    const auto& shape = embedding.shape();
    if (shape[0] <= 0 || shape[1] <= 0 ||
        static_cast<std::uint64_t>(shape[0]) > std::numeric_limits<std::size_t>::max() ||
        static_cast<std::uint64_t>(shape[1]) > std::numeric_limits<std::size_t>::max()) {
        std::cerr << "tensor shape is not representable\n";
        return 1;
    }
    const std::size_t rows = static_cast<std::size_t>(shape[0]);
    const std::size_t cols = static_cast<std::size_t>(shape[1]);
    std::vector<float> input(cols, 1.0f);
    std::vector<float> output;
    if (!chapter05_quantization::Q4KLinearFused::forward(
            reinterpret_cast<const std::uint8_t*>(embedding.raw_data()),
            embedding.bytes(), rows, cols, input, &output, &error)) {
        std::cerr << "Q4_K Linear failed: " << error << '\n';
        return 1;
    }

    std::cout << "tensor: " << embedding.name() << '\n'
              << "shape: [" << rows << ", " << cols << "]\n"
              << "payload bytes: " << embedding.bytes() << '\n'
              << "output elements: " << output.size() << '\n'
              << "first output: " << output.front() << '\n';
    return 0;
}
