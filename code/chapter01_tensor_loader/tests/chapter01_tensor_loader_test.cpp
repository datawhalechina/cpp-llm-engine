// Chapter 01：模型加载与 schema 综合测试
// 职责：验证 CIE1 loader、GGUF index、metadata、tokenizer、Q8_0/Q4_K 和 DecoderInput。
// 通过条件：fixture、非法输入和失败原子性检查全部成立。
// 验证：bash code/chapter01_tensor_loader/run.sh。
#include "../src/model_loader.h"
#include "../src/model_config.h"
#include "../src/model_metadata.h"
#include "../src/model_metadata_adapter.h"
#include "../src/gguf_reader.h"
#include "../src/gguf_tensor_loader.h"
#include "../src/model_builder.h"
#include "../src/tokenizer_metadata.h"
#include "../src/tensor.h"
#include "../src/tokenizer.h"
#include "cpp_llm_engine/testing/test.h"
#include "support/chapter01_test_groups.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using chapter01::DType;
using chapter01::Device;
using chapter01::ModelLoader;
using chapter01::ModelConfig;
using chapter01::ModelMetadata;
using chapter01::MetadataMap;
using chapter01::ModelMetadataAdapter;
using chapter01::GgufReader;
using chapter01::GgufModelIndex;
using chapter01::GgufTensorLoader;
using chapter01::LoadedModel;
using chapter01::ModelBuilder;
using chapter01::TokenizerMetadata;
using chapter01::TokenizerMetadataAdapter;
using chapter01::Tensor;
using chapter01::Tokenizer;
using cpp_llm_engine::testing::TestSuite;

namespace {

constexpr char kMagic[4] = {'C', 'I', 'E', '1'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_string(std::ostream& out, const std::string& value) {
    const std::uint32_t size = static_cast<std::uint32_t>(value.size());
    write_pod(out, size);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_gguf_string(std::ostream& out, const std::string& value) {
    const auto size = static_cast<std::uint64_t>(value.size());
    write_pod(out, size);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_gguf_kv_string(std::ostream& out, const std::string& key, const std::string& value) {
    write_gguf_string(out, key);
    const std::uint32_t type = 8;
    write_pod(out, type);
    write_gguf_string(out, value);
}

void write_gguf_kv_u32(std::ostream& out, const std::string& key, std::uint32_t value) {
    write_gguf_string(out, key);
    const std::uint32_t type = 4;
    write_pod(out, type);
    write_pod(out, value);
}

template <typename T>
void write_gguf_array(std::ostream& out, const std::string& key, std::uint32_t element_type, const std::vector<T>& values) {
    write_gguf_string(out, key);
    const std::uint32_t array_type = 9;
    write_pod(out, array_type);
    write_pod(out, element_type);
    const auto count = static_cast<std::uint64_t>(values.size());
    write_pod(out, count);
    for (const auto& value : values) {
        if constexpr (std::is_same_v<T, std::string>) {
            write_gguf_string(out, value);
        } else {
            write_pod(out, value);
        }
    }
}

void create_mock_gguf(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to create mock GGUF: " + path.string());
    out.write("GGUF", 4);
    const std::uint32_t version = 3;
    const std::uint64_t tensor_count = 1;
    const std::uint64_t metadata_count = 11;
    write_pod(out, version);
    write_pod(out, tensor_count);
    write_pod(out, metadata_count);
    write_gguf_kv_string(out, "general.architecture", "llama");
    write_gguf_kv_u32(out, "tokenizer.ggml.vocab_size", 64);
    write_gguf_kv_u32(out, "llama.embedding_length", 16);
    write_gguf_kv_u32(out, "llama.context_length", 32);
    write_gguf_kv_u32(out, "llama.block_count", 2);
    write_gguf_kv_u32(out, "tokenizer.ggml.bos_token_id", 1);
    write_gguf_kv_u32(out, "tokenizer.ggml.eos_token_id", 2);
    std::vector<std::string> tokens(64);
    std::vector<float> scores(64, -1.0f);
    std::vector<std::uint32_t> token_types(64, 1);
    tokens[0] = "a";
    tokens[1] = "b";
    tokens[2] = "<eos>";
    token_types[2] = 3;
    write_gguf_array(out, "tokenizer.ggml.tokens", 8, tokens);
    write_gguf_array(out, "tokenizer.ggml.scores", 6, scores);
    write_gguf_array(out, "tokenizer.ggml.token_type", 4, token_types);
    write_gguf_kv_u32(out, "general.alignment", 32);
    write_gguf_string(out, "token_embd.weight");
    const std::uint32_t dimension_count = 2;
    const std::uint64_t dimension0 = 16;
    const std::uint64_t dimension1 = 64;
    const std::uint32_t type = 0;
    const std::uint64_t offset = 0;
    write_pod(out, dimension_count);
    write_pod(out, dimension0);
    write_pod(out, dimension1);
    write_pod(out, type);
    write_pod(out, offset);
    const auto position = static_cast<std::uint64_t>(out.tellp());
    const auto padding = (32 - (position % 32)) % 32;
    for (std::uint64_t i = 0; i < padding; ++i) {
        const char zero = 0;
        out.write(&zero, 1);
    }
    const std::vector<float> payload(64 * 16, 3.5f);
    out.write(reinterpret_cast<const char*>(payload.data()),
              static_cast<std::streamsize>(payload.size() * sizeof(float)));
}

void write_shape(std::ostream& out, const std::vector<std::int64_t>& shape) {
    const std::uint32_t ndim = static_cast<std::uint32_t>(shape.size());
    write_pod(out, ndim);
    for (auto dim : shape) {
        write_pod(out, dim);
    }
}

template <typename T>
std::vector<std::byte> to_bytes(const std::vector<T>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

void write_tensor_record(std::ostream& out,
                         const std::string& name,
                         DType dtype,
                         Device device,
                         const std::vector<std::int64_t>& shape,
                         const std::vector<std::byte>& bytes) {
    write_string(out, name);
    const std::uint8_t dtype_u8 = static_cast<std::uint8_t>(dtype);
    const std::uint8_t device_u8 = static_cast<std::uint8_t>(device);
    write_pod(out, dtype_u8);
    write_pod(out, device_u8);
    write_shape(out, shape);
    const std::uint64_t byte_count = static_cast<std::uint64_t>(bytes.size());
    write_pod(out, byte_count);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

struct TeachingTensorRecord {
    std::string name;
    DType dtype;
    Device device;
    std::vector<std::int64_t> shape;
    std::vector<std::byte> bytes;
};

void create_teaching_model(const std::filesystem::path& path,
                           const std::vector<TeachingTensorRecord>& records) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to create teaching model: " + path.string());
    out.write(kMagic, sizeof(kMagic));
    write_pod(out, kVersion);
    const auto tensor_count = static_cast<std::uint32_t>(records.size());
    write_pod(out, tensor_count);
    for (const auto& record : records) {
        write_tensor_record(
            out, record.name, record.dtype, record.device, record.shape, record.bytes);
    }
}

std::filesystem::path make_temp_model_path() {
    auto path = std::filesystem::temp_directory_path() / "chapter01_tensor_loader_mock.bin";
    std::filesystem::remove(path);
    return path;
}

void create_mock_model(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to create mock model: " + path.string());

    out.write(kMagic, sizeof(kMagic));
    write_pod(out, kVersion);

    const std::uint32_t tensor_count = 2;
    write_pod(out, tensor_count);

    {
        const std::vector<float> values = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        write_tensor_record(out,
                            "linear.weight",
                            DType::Float32,
                            Device::CPU,
                            {2, 3},
                            to_bytes(values));
    }

    {
        const std::vector<std::int8_t> values = {-1, 0, 1, 2};
        write_tensor_record(out,
                            "quant.weight",
                            DType::Int8,
                            Device::CPU,
                            {4},
                            to_bytes(values));
    }
}

void test_tensor_roundtrip(TestSuite& suite) {
    Tensor tensor = Tensor::from_vector<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
    CPP_LLM_CHECK(suite, tensor.name() == "x");
    CPP_LLM_CHECK(suite, tensor.dtype() == DType::Float32);
    CPP_LLM_CHECK(suite, tensor.numel() == 4);
    const auto values = tensor.to_vector<float>();
    CPP_LLM_REQUIRE(suite, values.size() == 4);
    CPP_LLM_CHECK(suite, values[0] == 1.f);
    CPP_LLM_CHECK(suite, values[3] == 4.f);

    bool rejected_type = false;
    try {
        (void)tensor.to_vector<std::int8_t>();
    } catch (const std::invalid_argument&) {
        rejected_type = true;
    }
    CPP_LLM_CHECK(suite, rejected_type);

    bool rejected_vector_size = false;
    try {
        (void)Tensor::from_vector<float>("bad-size", {2}, {1.0f});
    } catch (const std::invalid_argument&) {
        rejected_vector_size = true;
    }
    CPP_LLM_CHECK(suite, rejected_vector_size);

    Tensor short_storage("short", {2}, DType::Float32);
    short_storage.set_storage(std::make_shared<Tensor::Storage>(sizeof(float)));
    bool rejected_short_storage = false;
    try {
        (void)short_storage.to_vector<float>();
    } catch (const std::out_of_range&) {
        rejected_short_storage = true;
    }
    CPP_LLM_CHECK(suite, rejected_short_storage);

    Tensor offset_tensor("offset", {1}, DType::Float32);
    bool rejected_offset = false;
    try {
        offset_tensor.set_storage(std::make_shared<Tensor::Storage>(1), 2);
    } catch (const std::invalid_argument&) {
        rejected_offset = true;
    }
    CPP_LLM_CHECK(suite, rejected_offset);

    Tensor incomplete_offset = Tensor::from_vector<float>(
        "incomplete-offset", {1}, {1.0f});
    bool rejected_incomplete_offset = false;
    try {
        incomplete_offset.set_offset_bytes(1);
    } catch (const std::invalid_argument&) {
        rejected_incomplete_offset = true;
    }
    CPP_LLM_CHECK(suite, rejected_incomplete_offset);

    Tensor oversized("oversized", {std::numeric_limits<std::int64_t>::max(), 3}, DType::Float32);
    bool rejected_shape = false;
    try {
        (void)oversized.numel();
    } catch (const std::overflow_error&) {
        rejected_shape = true;
    }
    CPP_LLM_CHECK(suite, rejected_shape);
}

void test_model_loader(TestSuite& suite) {
    const auto path = make_temp_model_path();
    create_mock_model(path);

    ModelLoader loader;
    std::string error;
    const bool ok = loader.load(path, &error);
    CPP_LLM_REQUIRE(suite, ok && error.empty());

    const Tensor* linear = loader.find("linear.weight");
    CPP_LLM_REQUIRE(suite, linear != nullptr);
    CPP_LLM_CHECK(suite, linear->dtype() == DType::Float32);
    CPP_LLM_REQUIRE(suite, linear->shape().size() == 2);
    CPP_LLM_CHECK(suite, linear->shape()[0] == 2);
    CPP_LLM_CHECK(suite, linear->shape()[1] == 3);
    const auto linear_values = linear->to_vector<float>();
    CPP_LLM_REQUIRE(suite, linear_values.size() == 6);
    CPP_LLM_CHECK(suite, linear_values[0] == 1.f);
    CPP_LLM_CHECK(suite, linear_values[5] == 6.f);

    const Tensor* quant = loader.find("quant.weight");
    CPP_LLM_REQUIRE(suite, quant != nullptr);
    CPP_LLM_CHECK(suite, quant->dtype() == DType::Int8);
    CPP_LLM_REQUIRE(suite, quant->shape().size() == 1);
    CPP_LLM_CHECK(suite, quant->shape()[0] == 4);
    const auto quant_values = quant->to_vector<std::int8_t>();
    CPP_LLM_CHECK(suite, (quant_values == std::vector<std::int8_t>{-1, 0, 1, 2}));

    std::filesystem::remove(path);
}

void test_model_loader_rejects_malformed_files(TestSuite& suite) {
    const auto path = make_temp_model_path();
    ModelLoader loader;
    std::string error;

    {
        std::ofstream out(path, std::ios::binary);
        CPP_LLM_REQUIRE(suite, out);
        out.write("BAD!", 4);
    }
    CPP_LLM_CHECK(suite, !loader.load(path, &error));
    CPP_LLM_CHECK(suite, error == "invalid model magic");
    CPP_LLM_CHECK(suite, loader.tensors().empty());

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        CPP_LLM_REQUIRE(suite, out);
        out.write(kMagic, sizeof(kMagic));
        const std::uint32_t unsupported_version = kVersion + 1;
        write_pod(out, unsupported_version);
    }
    CPP_LLM_CHECK(suite, !loader.load(path, &error));
    CPP_LLM_CHECK(suite, error == "unsupported model version");
    CPP_LLM_CHECK(suite, loader.tensors().empty());

    create_mock_model(path);
    const auto complete_size = std::filesystem::file_size(path);
    CPP_LLM_REQUIRE(suite, complete_size > 0);
    std::filesystem::resize_file(path, complete_size - 1);
    CPP_LLM_CHECK(suite, !loader.load(path, &error));
    CPP_LLM_CHECK(suite, error == "failed to read tensor payload");
    CPP_LLM_CHECK(suite, loader.tensors().empty());

    create_teaching_model(
        path,
        {{"unknown", static_cast<DType>(255), Device::CPU, {1}, {std::byte{0}}}});
    CPP_LLM_CHECK(suite, !loader.load(path, &error));
    CPP_LLM_CHECK(suite, error == "unsupported teaching tensor dtype");
    CPP_LLM_CHECK(suite, loader.tensors().empty());

    create_teaching_model(
        path,
        {{"device", DType::Float32, static_cast<Device>(1), {1}, to_bytes(std::vector<float>{1.0f})}});
    CPP_LLM_CHECK(suite, !loader.load(path, &error));
    CPP_LLM_CHECK(suite, error == "unsupported teaching tensor device");
    CPP_LLM_CHECK(suite, loader.tensors().empty());

    create_teaching_model(
        path,
        {{"shape", DType::Float32, Device::CPU, {-1}, to_bytes(std::vector<float>{1.0f})}});
    CPP_LLM_CHECK(suite, !loader.load(path, &error));
    CPP_LLM_CHECK(suite, error == "invalid teaching tensor shape");
    CPP_LLM_CHECK(suite, loader.tensors().empty());

    create_teaching_model(
        path,
        {{"payload", DType::Float32, Device::CPU, {2}, to_bytes(std::vector<float>{1.0f})}});
    CPP_LLM_CHECK(suite, !loader.load(path, &error));
    CPP_LLM_CHECK(
        suite, error == "teaching tensor payload size does not match shape and dtype");
    CPP_LLM_CHECK(suite, loader.tensors().empty());

    const TeachingTensorRecord duplicate{
        "duplicate", DType::Float32, Device::CPU, {1}, to_bytes(std::vector<float>{1.0f})};
    create_teaching_model(path, {duplicate, duplicate});
    CPP_LLM_CHECK(suite, !loader.load(path, &error));
    CPP_LLM_CHECK(suite, error == "duplicate teaching tensor name");
    CPP_LLM_CHECK(suite, loader.tensors().empty());
    std::filesystem::remove(path);
}

void test_tokenizer_contract(TestSuite& suite) {
    const Tokenizer tokenizer(64);
    const auto tokens = tokenizer.encode("a!");
    CPP_LLM_CHECK(suite, tokenizer.vocab_size() == 64);
    CPP_LLM_REQUIRE(suite, tokens.size() == 2);
    CPP_LLM_CHECK(suite, tokens[0] == 1);
    CPP_LLM_CHECK(suite, tokens[1] == 0);
    CPP_LLM_CHECK(suite, tokenizer.decode(tokens) == "aa");
}

void test_model_config_contract(TestSuite& suite) {
    ModelConfig config;
    Tokenizer tokenizer(config.vocab_size);
    std::string error;
    CPP_LLM_CHECK(suite, config.valid());
    CPP_LLM_CHECK(suite, config.validate_tokenizer(tokenizer, &error));
    CPP_LLM_CHECK(suite, error.empty());

    Tensor embedding = Tensor::from_vector<float>(
        "embedding.weight", {64, 16}, std::vector<float>(64 * 16, 1.0f));
    CPP_LLM_CHECK(suite, config.validate_tensor(embedding, &error));

    Tensor wrong_embedding = Tensor::from_vector<float>(
        "embedding.weight", {32, 16}, std::vector<float>(32 * 16, 1.0f));
    CPP_LLM_CHECK(suite, !config.validate_tensor(wrong_embedding, &error));
    CPP_LLM_CHECK(suite, !error.empty());
}

void test_model_metadata_boundary(TestSuite& suite) {
    ModelMetadata metadata;
    metadata.architecture = "llama";
    metadata.vocab_size = 64;
    metadata.hidden_size = 16;
    metadata.max_position = 32;
    metadata.layer_count = 2;
    metadata.bos_token_id = 1;
    metadata.eos_token_id = 2;

    CPP_LLM_CHECK(suite, metadata.valid());
    const ModelConfig config = ModelConfig::from_metadata(metadata);
    CPP_LLM_CHECK(suite, config.architecture == "llama");
    CPP_LLM_CHECK(suite, config.vocab_size == 64);
    CPP_LLM_CHECK(suite, config.hidden_size == 16);
    CPP_LLM_CHECK(suite, config.bos_token_id == 1);
    CPP_LLM_CHECK(suite, config.eos_token_id == 2);
}

void test_gguf_metadata_adapter(TestSuite& suite) {
    const MetadataMap values = {
        {"general.architecture", std::string("llama")},
        {"tokenizer.ggml.vocab_size", std::uint64_t{32000}},
        {"llama.embedding_length", std::uint64_t{2048}},
        {"llama.context_length", std::uint64_t{2048}},
        {"llama.block_count", std::uint64_t{22}},
        {"tokenizer.ggml.bos_token_id", std::int64_t{1}},
        {"tokenizer.ggml.eos_token_id", std::int64_t{2}},
    };

    ModelMetadata metadata;
    std::string error;
    CPP_LLM_REQUIRE(suite, ModelMetadataAdapter::from_gguf(values, &metadata, &error));
    CPP_LLM_CHECK(suite, error.empty());
    CPP_LLM_CHECK(suite, metadata.architecture == "llama");
    CPP_LLM_CHECK(suite, metadata.vocab_size == 32000);
    CPP_LLM_CHECK(suite, metadata.hidden_size == 2048);
    CPP_LLM_CHECK(suite, metadata.max_position == 2048);
    CPP_LLM_CHECK(suite, metadata.layer_count == 22);
    CPP_LLM_CHECK(suite, metadata.bos_token_id == 1);
    CPP_LLM_CHECK(suite, metadata.eos_token_id == 2);

    MetadataMap incomplete = values;
    incomplete.erase("llama.block_count");
    CPP_LLM_CHECK(suite, !ModelMetadataAdapter::from_gguf(incomplete, &metadata, &error));
    CPP_LLM_CHECK(suite, error.find("block_count") != std::string::npos);

    MetadataMap inferred_vocab = values;
    inferred_vocab.erase("tokenizer.ggml.vocab_size");
    inferred_vocab["tokenizer.ggml.tokens"] = std::vector<std::string>(64, "token");
    CPP_LLM_REQUIRE(
        suite, ModelMetadataAdapter::from_gguf(inferred_vocab, &metadata, &error));
    CPP_LLM_CHECK(suite, metadata.vocab_size == 64);

    MetadataMap signed_token_types = values;
    signed_token_types["tokenizer.ggml.tokens"] = std::vector<std::string>(64, "token");
    signed_token_types["tokenizer.ggml.scores"] = std::vector<float>(64, -1.0f);
    signed_token_types["tokenizer.ggml.token_type"] = std::vector<std::int32_t>(64, 1);
    TokenizerMetadata tokenizer_metadata;
    CPP_LLM_REQUIRE(
        suite,
        TokenizerMetadataAdapter::from_gguf(
            signed_token_types, &tokenizer_metadata, &error));
}

void test_gguf_reader(TestSuite& suite) {
    const auto path = std::filesystem::temp_directory_path() / "chapter01_mock.gguf";
    std::filesystem::remove(path);
    create_mock_gguf(path);

    GgufModelIndex index;
    std::string error;
    const bool index_loaded = GgufReader::read_index(path.string(), &index, &error);
    if (!index_loaded) throw std::runtime_error(error);
    CPP_LLM_CHECK(suite, index_loaded);
    CPP_LLM_CHECK(suite, error.empty());
    CPP_LLM_CHECK(suite, index.version == 3);
    CPP_LLM_CHECK(suite, std::get<std::string>(index.metadata.at("general.architecture")) == "llama");
    CPP_LLM_REQUIRE(suite, index.tensors.size() == 1);
    CPP_LLM_CHECK(suite, index.tensors[0].name == "token_embd.weight");
    CPP_LLM_CHECK(suite, index.tensors[0].dimensions == std::vector<std::uint64_t>({16, 64}));
    CPP_LLM_CHECK(suite, index.tensors[0].type == 0);
    CPP_LLM_CHECK(suite, index.tensors[0].offset == 0);
    CPP_LLM_CHECK(suite, index.data_offset % 32 == 0);

    ModelMetadata metadata;
    CPP_LLM_REQUIRE(
        suite, ModelMetadataAdapter::from_gguf(index.metadata, &metadata, &error));
    CPP_LLM_CHECK(suite, metadata.valid());
    CPP_LLM_CHECK(suite, metadata.architecture == "llama");
    CPP_LLM_CHECK(suite, metadata.vocab_size == 64);
    CPP_LLM_CHECK(suite, metadata.hidden_size == 16);
    CPP_LLM_CHECK(suite, metadata.max_position == 32);
    CPP_LLM_CHECK(suite, metadata.layer_count == 2);
    CPP_LLM_CHECK(suite, metadata.bos_token_id == 1);
    CPP_LLM_CHECK(suite, metadata.eos_token_id == 2);

    TokenizerMetadata tokenizer_metadata;
    CPP_LLM_REQUIRE(
        suite,
        TokenizerMetadataAdapter::from_gguf(index.metadata, &tokenizer_metadata, &error));
    CPP_LLM_CHECK(suite, tokenizer_metadata.valid());
    CPP_LLM_REQUIRE(suite, tokenizer_metadata.tokens.size() == 64);
    CPP_LLM_CHECK(suite, tokenizer_metadata.tokens[2] == "<eos>");
    CPP_LLM_CHECK(suite, tokenizer_metadata.scores.size() == 64);
    CPP_LLM_CHECK(suite, tokenizer_metadata.token_types.size() == 64);

    std::unordered_map<std::string, Tensor> tensors;
    const bool tensors_loaded = GgufTensorLoader::load_all(path, index, &tensors, &error);
    if (!tensors_loaded) throw std::runtime_error(error);
    CPP_LLM_CHECK(suite, tensors_loaded);
    const Tensor* embedding = tensors.at("token_embd.weight").name().empty()
                                  ? nullptr
                                  : &tensors.at("token_embd.weight");
    CPP_LLM_REQUIRE(suite, embedding != nullptr);
    CPP_LLM_CHECK(suite, embedding->dtype() == DType::Float32);
    CPP_LLM_CHECK(suite, embedding->bytes() == 64 * 16 * sizeof(float));
    CPP_LLM_CHECK(suite, embedding->to_vector<float>()[0] == 3.5f);

    auto add_tensor = [&tensors](const std::string& name, std::vector<std::int64_t> shape) {
        std::size_t elements = 1;
        for (const auto dimension : shape) elements *= static_cast<std::size_t>(dimension);
        tensors.emplace(name, Tensor::from_vector<float>(name, std::move(shape), std::vector<float>(elements, 1.0f)));
    };
    add_tensor("output.weight", {16, 64});
    add_tensor("output_norm.weight", {16});
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        add_tensor(prefix + "attn_norm.weight", {16});
        add_tensor(prefix + "attn_q.weight", {16, 16});
        add_tensor(prefix + "attn_k.weight", {16, 8});
        add_tensor(prefix + "attn_v.weight", {16, 8});
        add_tensor(prefix + "attn_output.weight", {16, 16});
        add_tensor(prefix + "ffn_norm.weight", {16});
        add_tensor(prefix + "ffn_gate.weight", {16, 32});
        add_tensor(prefix + "ffn_up.weight", {16, 32});
        add_tensor(prefix + "ffn_down.weight", {32, 16});
    }

    LoadedModel model;
    model.config.architecture = "preserved";
    std::unordered_map<std::string, Tensor> incomplete_tensors = tensors;
    incomplete_tensors.erase("output.weight");
    CPP_LLM_CHECK(suite, !ModelBuilder::build(metadata, tokenizer_metadata, std::move(incomplete_tensors), &model, &error));
    CPP_LLM_CHECK(suite, error.find("missing Llama tensor") != std::string::npos);
    CPP_LLM_CHECK(suite, model.config.architecture == "preserved");

    std::unordered_map<std::string, Tensor> wrong_shape_tensors = tensors;
    wrong_shape_tensors["output_norm.weight"] = Tensor::from_vector<float>(
        "output_norm.weight", {15}, std::vector<float>(15, 1.0f));
    CPP_LLM_CHECK(
        suite,
        !ModelBuilder::build(
            metadata, tokenizer_metadata, std::move(wrong_shape_tensors), &model, &error));
    CPP_LLM_CHECK(suite, error == "shape mismatch for output_norm.weight");
    CPP_LLM_CHECK(suite, model.config.architecture == "preserved");

    CPP_LLM_REQUIRE(
        suite,
        ModelBuilder::build(
            metadata, tokenizer_metadata, std::move(tensors), &model, &error));
    CPP_LLM_CHECK(suite, error.empty());
    CPP_LLM_CHECK(suite, model.config.architecture == "llama");
    CPP_LLM_CHECK(suite, model.tokenizer.tokens.size() == 64);
    CPP_LLM_CHECK(suite, model.layout.layers.size() == 2);
    CPP_LLM_CHECK(suite, model.layout.intermediate_size == 32);
    CPP_LLM_CHECK(suite, model.layout.key_value_size == 8);
    CPP_LLM_CHECK(suite, model.decoder.valid());
    CPP_LLM_CHECK(suite, model.decoder.embedding == "token_embd.weight");
    CPP_LLM_REQUIRE(suite, model.decoder.layers.size() == 2);
    CPP_LLM_CHECK(suite, model.decoder.layers[0].attn_q == "blk.0.attn_q.weight");
    CPP_LLM_CHECK(suite, model.decoder.layers[1].ffn_down == "blk.1.ffn_down.weight");
    CPP_LLM_CHECK(suite, model.tensors.size() == 21);
    std::filesystem::remove(path);

    GgufModelIndex preserved;
    preserved.version = 77;
    {
        std::ofstream out(path, std::ios::binary);
        CPP_LLM_REQUIRE(suite, out);
        out.write("GGUF", 4);
        const std::uint32_t version = 3;
        const std::uint64_t tensor_count = (1U << 20U) + 1U;
        const std::uint64_t metadata_count = 0;
        write_pod(out, version);
        write_pod(out, tensor_count);
        write_pod(out, metadata_count);
    }
    CPP_LLM_CHECK(suite, !GgufReader::read_index(path.string(), &preserved, &error));
    CPP_LLM_CHECK(suite, error == "GGUF header count exceeds teaching reader limit");
    CPP_LLM_CHECK(suite, preserved.version == 77);

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        CPP_LLM_REQUIRE(suite, out);
        out.write("GGUF", 4);
        const std::uint32_t version = 3;
        const std::uint64_t tensor_count = 1;
        const std::uint64_t metadata_count = 0;
        write_pod(out, version);
        write_pod(out, tensor_count);
        write_pod(out, metadata_count);
        write_gguf_string(out, "bad-rank");
        const std::uint32_t dimension_count = 9;
        write_pod(out, dimension_count);
    }
    CPP_LLM_CHECK(suite, !GgufReader::read_index(path.string(), &preserved, &error));
    CPP_LLM_CHECK(suite, error == "GGUF tensor rank exceeds teaching reader limit");
    CPP_LLM_CHECK(suite, preserved.version == 77);

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        CPP_LLM_REQUIRE(suite, out);
        out.write("GGUF", 4);
        const std::uint32_t version = 3;
        const std::uint64_t tensor_count = 0;
        const std::uint64_t metadata_count = 2;
        write_pod(out, version);
        write_pod(out, tensor_count);
        write_pod(out, metadata_count);
        write_gguf_kv_u32(out, "duplicate.key", 1);
        write_gguf_kv_u32(out, "duplicate.key", 2);
    }
    CPP_LLM_CHECK(suite, !GgufReader::read_index(path.string(), &preserved, &error));
    CPP_LLM_CHECK(suite, error == "duplicate GGUF metadata key: duplicate.key");
    CPP_LLM_CHECK(suite, preserved.version == 77);

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        CPP_LLM_REQUIRE(suite, out);
        out.write("GGUF", 4);
        const std::uint32_t version = 3;
        const std::uint64_t tensor_count = 2;
        const std::uint64_t metadata_count = 0;
        write_pod(out, version);
        write_pod(out, tensor_count);
        write_pod(out, metadata_count);
        for (int i = 0; i < 2; ++i) {
            write_gguf_string(out, "duplicate.weight");
            const std::uint32_t dimension_count = 1;
            const std::uint64_t dimension = 1;
            const std::uint32_t type = 0;
            const std::uint64_t offset = 0;
            write_pod(out, dimension_count);
            write_pod(out, dimension);
            write_pod(out, type);
            write_pod(out, offset);
        }
    }
    CPP_LLM_CHECK(suite, !GgufReader::read_index(path.string(), &preserved, &error));
    CPP_LLM_CHECK(suite, error == "duplicate GGUF tensor name: duplicate.weight");
    CPP_LLM_CHECK(suite, preserved.version == 77);
    std::filesystem::remove(path);
}

void test_q4_k_payload_loader(TestSuite& suite) {
    const auto path = std::filesystem::temp_directory_path() / "chapter01_q4_k.gguf";
    std::filesystem::remove(path);
    {
        std::ofstream out(path, std::ios::binary);
        CPP_LLM_REQUIRE(suite, out);
        std::vector<std::byte> payload(144, std::byte{0x2a});
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
    }

    GgufModelIndex index;
    index.data_offset = 0;
    index.tensors.push_back({"q4.weight", {256}, 12, 0});
    std::unordered_map<std::string, Tensor> tensors;
    std::string error;
    const bool loaded = GgufTensorLoader::load_all(path, index, &tensors, &error);
    if (!loaded) throw std::runtime_error(error);
    CPP_LLM_REQUIRE(suite, loaded);
    const Tensor& tensor = tensors.at("q4.weight");
    CPP_LLM_CHECK(suite, tensor.dtype() == DType::Q4_K);
    CPP_LLM_CHECK(suite, tensor.numel() == 256);
    CPP_LLM_CHECK(suite, tensor.bytes() == 144);
    CPP_LLM_CHECK(suite, tensor.raw_data()[0] == std::byte{0x2a});

    GgufModelIndex invalid_index = index;
    invalid_index.tensors.push_back({"unsupported.weight", {1}, 999, 0});
    std::unordered_map<std::string, Tensor> unchanged;
    unchanged.emplace(
        "sentinel", Tensor::from_vector<float>("sentinel", {1}, {1.0f}));
    CPP_LLM_CHECK(
        suite, !GgufTensorLoader::load_all(path, invalid_index, &unchanged, &error));
    CPP_LLM_CHECK(suite, unchanged.size() == 1);
    CPP_LLM_CHECK(suite, unchanged.find("sentinel") != unchanged.end());
    std::filesystem::remove(path);
}

void test_q8_0_payload_loader(TestSuite& suite) {
    const auto path = std::filesystem::temp_directory_path() / "chapter01_q8_0.gguf";
    std::filesystem::remove(path);
    {
        std::ofstream out(path, std::ios::binary);
        CPP_LLM_REQUIRE(suite, out);
        std::vector<std::byte> payload(34, std::byte{0x17});
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
    }

    GgufModelIndex index;
    index.data_offset = 0;
    index.tensors.push_back({"q8.weight", {32}, 8, 0});
    Tensor tensor;
    std::string error;
    const bool loaded = GgufTensorLoader::load_tensor(
        path, index, index.tensors.front(), &tensor, &error);
    if (!loaded) throw std::runtime_error(error);
    CPP_LLM_REQUIRE(suite, loaded);
    CPP_LLM_CHECK(suite, error.empty());
    CPP_LLM_CHECK(suite, tensor.dtype() == DType::Q8_0);
    CPP_LLM_CHECK(suite, tensor.numel() == 32);
    CPP_LLM_CHECK(suite, tensor.bytes() == 34);
    CPP_LLM_CHECK(suite, tensor.raw_data()[33] == std::byte{0x17});

    index.tensors.front().dimensions = {31};
    const bool rejected = !GgufTensorLoader::load_tensor(
        path, index, index.tensors.front(), &tensor, &error);
    if (!rejected) throw std::runtime_error("misaligned Q8_0 tensor was accepted");
    CPP_LLM_CHECK(suite, rejected);
    CPP_LLM_CHECK(suite, error == "GGUF quantized tensor shape is not block aligned");
    std::filesystem::remove(path);
}

}  // namespace

int run_chapter01_test_group(const std::string& group) {
    const auto selected = [&group](const char* name) {
        return group == "all" || group == name;
    };
    if (!selected("loader") && !selected("schema") && !selected("tokenizer_payload")) {
        std::cerr << "unknown Chapter01 test group: " << group << '\n';
        return 2;
    }

    TestSuite suite("chapter01_" + group);
    if (selected("loader")) {
        suite.run("tensor roundtrip rejects invalid storage and shape", [&] {
            test_tensor_roundtrip(suite);
        });
        suite.run("teaching model loader restores named tensors", [&] {
            test_model_loader(suite);
        });
        suite.run("teaching model loader rejects malformed files atomically", [&] {
            test_model_loader_rejects_malformed_files(suite);
        });
    }
    if (selected("schema")) {
        suite.run("model config validates tokenizer and embedding shape", [&] {
            test_model_config_contract(suite);
        });
        suite.run("model metadata becomes stable internal config", [&] {
            test_model_metadata_boundary(suite);
        });
        suite.run("GGUF metadata adapter reports missing fields", [&] {
            test_gguf_metadata_adapter(suite);
        });
        suite.run("GGUF reader builds tokenizer tensors and decoder schema", [&] {
            test_gguf_reader(suite);
        });
    }
    if (selected("tokenizer_payload")) {
        suite.run("toy tokenizer preserves interface contract", [&] {
            test_tokenizer_contract(suite);
        });
        suite.run("Q8_0 loader rejects non-block-aligned shape", [&] {
            test_q8_0_payload_loader(suite);
        });
        suite.run("Q4_K loader preserves quantized payload", [&] {
            test_q4_k_payload_loader(suite);
        });
    }
    return suite.finish();
}
