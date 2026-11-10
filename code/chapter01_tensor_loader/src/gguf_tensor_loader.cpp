// Chapter 01：GGUF tensor payload loader 实现
// 职责：根据 GGUF tensor index 加载指定 tensor 的 raw bytes 和 dtype 信息。
// 契约：shape、dtype、payload byte 数必须与 index 一致。
// 验证：chapter01_tensor_loader_test.cpp 和 --load-first-q4。
#include "gguf_tensor_loader.h"

#include <fstream>
#include <limits>
#include <sstream>

namespace chapter01 {
namespace {

struct GgufDType {
    DType tensor_type = DType::Unknown;
    std::size_t element_bytes = 0;
    std::size_t block_elements = 1;
    std::size_t block_bytes = 0;
};

bool set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

bool map_dtype(std::uint32_t gguf_type, GgufDType* dtype, std::string* error) {
    // GGUF 类型编号沿用 ggml 定义，内部只暴露教程使用的 DType。
    switch (gguf_type) {
    case 0:
        *dtype = {DType::Float32, sizeof(float)};
        return true;
    case 1:
        *dtype = {DType::Float16, sizeof(std::uint16_t)};
        return true;
    case 8:
        // Q8_0 每个 34 字节块保存一个 FP16 scale 和 32 个 int8 值。
        *dtype = {DType::Q8_0, 0, 32, 34};
        return true;
    case 10:
        // Q2_K 每个 84 字节 super-block 表示 256 个值。
        *dtype = {DType::Q2_K, 0, 256, 84};
        return true;
    case 7:
        // Q5_1 每个 24 字节块表示 32 个值。
        *dtype = {DType::Q5_1, 0, 32, 24};
        return true;
    case 6:
        // Q5_0 每个 22 字节块表示 32 个有符号 5-bit 值。
        *dtype = {DType::Q5_0, 0, 32, 22};
        return true;
    case 11:
        // Q3_K 每个 110 字节 super-block 表示 256 个值。
        *dtype = {DType::Q3_K, 0, 256, 110};
        return true;
    case 13:
        *dtype = {DType::Q5_K, 0, 256, 176};
        return true;
    case 24:
        *dtype = {DType::Int8, sizeof(std::int8_t)};
        return true;
    case 12:
        // Q4_K 每个 144 字节 super-block 表示 256 个值。
        *dtype = {DType::Q4_K, 0, 256, 144};
        return true;
    case 14:
        // Q6_K 每个 210 字节 super-block 表示 256 个值。
        *dtype = {DType::Q6_K, 0, 256, 210};
        return true;
    case 20:
        // IQ4_NL 每个 18 字节非线性 block 表示 32 个值。
        *dtype = {DType::IQ4_NL, 0, 32, 18};
        return true;
    default:
        return set_error(error, "GGUF dtype is not supported by the Chapter 01 loader: " +
                                   std::to_string(gguf_type));
    }
}

bool checked_product(const std::vector<std::uint64_t>& dimensions,
                     std::size_t* elements,
                     std::string* error) {
    if (dimensions.empty()) {
        return set_error(error, "GGUF tensor shape must not be empty");
    }
    std::size_t result = 1;
    for (const auto dimension : dimensions) {
        if (dimension == 0 || dimension > std::numeric_limits<std::size_t>::max() ||
            result > std::numeric_limits<std::size_t>::max() /
                         static_cast<std::size_t>(dimension)) {
            return set_error(error, "GGUF tensor shape is invalid or too large");
        }
        result *= static_cast<std::size_t>(dimension);
    }
    *elements = result;
    return true;
}

bool checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t* result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    *result = left + right;
    return true;
}

}  // namespace

bool GgufTensorLoader::load_tensor(const std::filesystem::path& path,
                                   const GgufModelIndex& index,
                                   const GgufTensorInfo& info,
                                   Tensor* tensor,
                                   std::string* error) {
    if (tensor == nullptr) {
        return set_error(error, "tensor output must not be null");
    }
    GgufDType dtype;
    if (!map_dtype(info.type, &dtype, error)) return false;

    std::size_t elements = 0;
    if (!checked_product(info.dimensions, &elements, error)) return false;
    std::size_t byte_count_host = 0;
    if (dtype.block_elements == 1) {
        if (elements > std::numeric_limits<std::size_t>::max() / dtype.element_bytes) {
            return set_error(error, "GGUF tensor payload is too large");
        }
        byte_count_host = elements * dtype.element_bytes;
    } else {
        if (elements == 0 || elements % dtype.block_elements != 0 ||
            elements / dtype.block_elements > std::numeric_limits<std::size_t>::max() / dtype.block_bytes) {
            return set_error(error, "GGUF quantized tensor shape is not block aligned");
        }
        byte_count_host = (elements / dtype.block_elements) * dtype.block_bytes;
    }
    const auto byte_count = static_cast<std::uint64_t>(byte_count_host);
    if (byte_count > static_cast<std::uint64_t>(
                         std::numeric_limits<std::streamsize>::max())) {
        return set_error(error, "GGUF tensor payload is not readable by this host");
    }
    std::uint64_t payload_offset = 0;
    if (!checked_add(index.data_offset, info.offset, &payload_offset)) {
        return set_error(error, "GGUF tensor offset overflows");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) return set_error(error, "unable to open GGUF file: " + path.string());
    input.seekg(0, std::ios::end);
    const auto file_end = input.tellg();
    if (file_end < 0 || payload_offset > static_cast<std::uint64_t>(file_end) ||
        byte_count > static_cast<std::uint64_t>(file_end) - payload_offset) {
        return set_error(error, "GGUF tensor payload exceeds file boundary: " + info.name);
    }
    if (payload_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return set_error(error, "GGUF tensor offset is not seekable");
    }
    input.seekg(static_cast<std::streamoff>(payload_offset), std::ios::beg);
    if (!input) return set_error(error, "failed to seek to GGUF tensor: " + info.name);

    std::vector<std::byte> bytes(byte_count_host);
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) return set_error(error, "failed to read GGUF tensor payload: " + info.name);
    }

    std::vector<std::int64_t> shape;
    shape.reserve(info.dimensions.size());
    for (const auto dimension : info.dimensions) {
        if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return set_error(error, "GGUF tensor dimension is not representable: " + info.name);
        }
        shape.push_back(static_cast<std::int64_t>(dimension));
    }
    *tensor = Tensor::from_raw_bytes(info.name, std::move(shape), dtype.tensor_type, std::move(bytes));
    if (error != nullptr) error->clear();
    return true;
}

bool GgufTensorLoader::load_all(const std::filesystem::path& path,
                                const GgufModelIndex& index,
                                std::unordered_map<std::string, Tensor>* tensors,
                                std::string* error) {
    if (tensors == nullptr) return set_error(error, "tensor map output must not be null");
    std::unordered_map<std::string, Tensor> loaded;
    loaded.reserve(index.tensors.size());
    for (const auto& info : index.tensors) {
        Tensor tensor;
        if (!load_tensor(path, index, info, &tensor, error)) return false;
        if (!loaded.emplace(info.name, std::move(tensor)).second) {
            return set_error(error, "duplicate GGUF tensor name: " + info.name);
        }
    }
    *tensors = std::move(loaded);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter01
