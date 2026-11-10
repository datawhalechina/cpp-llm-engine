// Chapter 01：GGUF inspection example
// 职责：读取 GGUF index、metadata、tokenizer 和 Llama tensor schema。
// 运行：gguf_inspect <model.gguf> [--load-first-q4|--validate-schema|--tensor <name>]。
// 验证：真实 TinyLlama schema 输出 decoder input: ready。
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"
#include "model_builder.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "tokenizer_metadata.h"

#include <iostream>

namespace {

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index,
    const std::string& name) {
    for (const auto& tensor : index.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

void print_tensor_anchor(const chapter01::GgufModelIndex& index,
                         const std::string& name) {
    const auto* tensor = find_tensor(index, name);
    if (tensor == nullptr) return;
    std::cout << name << " type: " << tensor->type << " shape:";
    for (const auto dimension : tensor->dimensions) std::cout << ' ' << dimension;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4 || (argc == 4 && std::string(argv[2]) != "--tensor")) {
        std::cerr << "usage: gguf_inspect <model.gguf> "
                     "[--load-first-q4|--validate-schema|--tensor <name>]\n";
        return 2;
    }

    chapter01::GgufModelIndex index;
    std::string error;
    if (!chapter01::GgufReader::read_index(argv[1], &index, &error)) {
        std::cerr << "read failed: " << error << "\n";
        return 1;
    }

    chapter01::ModelMetadata metadata;
    if (!chapter01::ModelMetadataAdapter::from_gguf(index.metadata, &metadata, &error)) {
        std::cerr << "metadata failed: " << error << "\n";
        return 1;
    }

    chapter01::TokenizerMetadata tokenizer;
    if (!chapter01::TokenizerMetadataAdapter::from_gguf(index.metadata, &tokenizer, &error)) {
        std::cerr << "tokenizer metadata failed: " << error << "\n";
        return 1;
    }

    std::cout << "architecture: " << metadata.architecture << "\n"
              << "vocab size: " << metadata.vocab_size << "\n"
              << "hidden size: " << metadata.hidden_size << "\n"
              << "context length: " << metadata.max_position << "\n"
              << "layers: " << metadata.layer_count << "\n"
              << "attention heads: " << metadata.attention_head_count << "\n"
              << "attention kv heads: " << metadata.attention_head_count_kv << "\n"
              << "attention key/value length: " << metadata.attention_key_length
              << "/" << metadata.attention_value_length << "\n"
              << "experts selected/total: " << metadata.expert_used_count
              << "/" << metadata.expert_count << "\n"
              << "special token ids: bos=" << metadata.bos_token_id
              << " eos=" << metadata.eos_token_id << "\n"
              << "chat template: "
              << (tokenizer.chat_template.empty() ? "no" : "yes") << "\n"
              << "tokenizer tokens: " << tokenizer.tokens.size() << "\n"
              << "tensors: " << index.tensors.size() << "\n"
              << "data offset: " << index.data_offset << "\n";

    if (argc == 4) {
        const std::string name = argv[3];
        if (find_tensor(index, name) == nullptr) {
            std::cerr << "tensor not found: " << name << "\n";
            return 1;
        }
        print_tensor_anchor(index, name);
        return 0;
    }

    if (argc == 3 && std::string(argv[2]) == "--validate-schema") {
        if (metadata.architecture == "qwen2moe") {
            chapter01::MoeDecoderTensorLayout layout;
            if (!chapter01::ModelSchema::validate_moe_gguf(
                    metadata, index, &layout, &error)) {
                std::cerr << "schema validation failed: " << error << "\n";
                return 1;
            }
            std::cout << "schema: qwen2moe tensors valid\n"
                      << "decoder layers: " << layout.layers.size() << "\n"
                      << "experts selected/total: " << layout.expert_used_count
                      << "/" << layout.expert_count << "\n"
                      << "decoder input: MoE schema ready\n";
            return 0;
        }
        chapter01::DenseDecoderTensorLayout layout;
        if (!chapter01::ModelSchema::validate_gguf(metadata, index, &layout, &error)) {
            std::cerr << "schema validation failed: " << error << "\n";
            return 1;
        }
        std::cout << "schema: " << metadata.architecture << " tensors valid\n"
                  << "decoder layers: " << layout.layers.size() << "\n"
                  << "intermediate size: " << layout.intermediate_size << "\n"
                  << "key/value size: " << layout.key_value_size << "\n"
                  << "output tied to embedding: "
                  << (layout.output_tied_to_embedding ? "yes" : "no") << "\n";
        const auto decoder = chapter01::DecoderInput::from_layout(layout);
        if (!decoder.valid()) {
            std::cerr << "decoder input assembly failed\n";
            return 1;
        }
        std::cout << "decoder input: ready\n"
                  << "first layer query: " << decoder.layers.front().attn_q << "\n"
                  << "last layer output: " << decoder.layers.back().attn_output << "\n";
        print_tensor_anchor(index, layout.output_norm);
        print_tensor_anchor(index, layout.output);
        return 0;
    }

    if (argc == 3 && std::string(argv[2]) == "--load-first-q4") {
        for (const auto& info : index.tensors) {
            if (info.type != 12) continue;
            chapter01::Tensor tensor;
            if (!chapter01::GgufTensorLoader::load_tensor(argv[1], index, info, &tensor, &error)) {
                std::cerr << "Q4_K load failed: " << error << "\n";
                return 1;
            }
            std::cout << "loaded Q4_K tensor: " << tensor.name()
                      << " elements=" << tensor.numel()
                      << " bytes=" << tensor.bytes() << "\n";
            return 0;
        }
        std::cerr << "no Q4_K tensor found\n";
        return 1;
    }
    return 0;
}
