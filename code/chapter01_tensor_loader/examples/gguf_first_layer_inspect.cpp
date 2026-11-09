// Chapter 01：检查真实 TinyLlama 第一层 decoder 权重装配。
// 职责：加载 layer-0 的九个 tensor，输出 dtype、shape 和 payload，不执行 forward。
// 运行：gguf_first_layer_inspect <model.gguf>；验证真实 GGUF 到 CPU layer 输入的边界。
#include "gguf_reader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"
#include "tokenizer_metadata.h"

#include <iostream>
#include <string>

namespace {

void print_tensor(const chapter01::Tensor& tensor) {
    std::cout << tensor.name() << " dtype=" << static_cast<int>(tensor.dtype()) << " shape=[";
    for (std::size_t index = 0; index < tensor.shape().size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << tensor.shape()[index];
    }
    std::cout << "] bytes=" << tensor.bytes() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: gguf_first_layer_inspect <model.gguf>\n";
        return 2;
    }
    chapter01::GgufModelIndex index;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error)) {
        std::cerr << "read failed: " << error << '\n';
        return 1;
    }
    chapter01::ModelMetadata metadata;
    if (!chapter01::ModelMetadataAdapter::from_gguf(index.metadata, &metadata, &error)) {
        std::cerr << "metadata failed: " << error << '\n';
        return 1;
    }
    if (metadata.architecture != "llama") {
        std::cerr << "first-layer inspector requires llama architecture\n";
        return 1;
    }
    chapter01::LlamaTensorLayout layout;
    if (!chapter01::ModelSchema::validate_gguf(metadata, index, &layout, &error)) {
        std::cerr << "schema failed: " << error << '\n';
        return 1;
    }
    chapter01::RealLlamaLayerWeights weights;
    if (!chapter01::RealLlamaLayerLoader::load(argv[1], index, layout, 0, &weights, &error)) {
        std::cerr << "layer load failed: " << error << '\n';
        return 1;
    }
    print_tensor(weights.attn_norm);
    print_tensor(weights.attn_q);
    print_tensor(weights.attn_k);
    print_tensor(weights.attn_v);
    print_tensor(weights.attn_output);
    print_tensor(weights.ffn_norm);
    print_tensor(weights.ffn_gate);
    print_tensor(weights.ffn_up);
    print_tensor(weights.ffn_down);
    std::cout << "first_layer_index=" << weights.index << '\n'
              << "first_layer_weight_contract=ready\n";
    return 0;
}
