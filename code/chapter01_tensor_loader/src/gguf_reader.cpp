// Chapter 01：GGUF index reader 实现
// 职责：读取 GGUF header、metadata、tensor index 和 data offset，不加载全部权重。
// 契约：文件格式错误、截断输入和不支持的 metadata 返回 error。
// 验证：chapter01_tensor_loader_test.cpp 与 gguf_inspect。
#include "gguf_reader.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <unordered_set>

namespace chapter01 {
namespace {

constexpr std::uint32_t kGgufVersionMin = 2;
constexpr std::uint32_t kGgufVersionMax = 3;
constexpr std::uint64_t kDefaultAlignment = 32;
constexpr std::uint64_t kMaxStringBytes = 64 * 1024 * 1024;
constexpr std::uint64_t kMaxArrayElements = 16 * 1024 * 1024;
constexpr std::uint64_t kMaxMetadataCount = 1U << 20U;
constexpr std::uint64_t kMaxTensorCount = 1U << 20U;
constexpr std::uint32_t kMaxTensorRank = 8;

enum class GgufType : std::uint32_t {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
};

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

template <typename T>
bool read_pod(std::istream& input, T* value) {
    input.read(reinterpret_cast<char*>(value), sizeof(T));
    return static_cast<bool>(input);
}

bool read_string(std::istream& input, std::string* value, std::string* error) {
    std::uint64_t size = 0;
    if (!read_pod(input, &size)) {
        set_error(error, "unexpected end of GGUF while reading string length");
        return false;
    }
    if (size > kMaxStringBytes || size > std::numeric_limits<std::size_t>::max()) {
        set_error(error, "GGUF string is too large");
        return false;
    }
    value->resize(static_cast<std::size_t>(size));
    if (size != 0) {
        input.read(value->data(), static_cast<std::streamsize>(size));
    }
    if (!input) {
        set_error(error, "unexpected end of GGUF while reading string");
        return false;
    }
    return true;
}

bool skip_bytes(std::istream& input, std::uint64_t bytes, std::string* error) {
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        set_error(error, "GGUF field is too large to skip");
        return false;
    }
    input.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!input) {
        set_error(error, "unexpected end of GGUF while skipping field");
        return false;
    }
    return true;
}

bool skip_value(std::istream& input, GgufType type, std::string* error);

bool skip_array(std::istream& input, std::string* error) {
    std::uint32_t element_type_raw = 0;
    std::uint64_t count = 0;
    if (!read_pod(input, &element_type_raw) || !read_pod(input, &count)) {
        set_error(error, "unexpected end of GGUF while reading array header");
        return false;
    }
    const auto element_type = static_cast<GgufType>(element_type_raw);
    if (count > kMaxArrayElements) {
        set_error(error, "GGUF metadata array is too large");
        return false;
    }
    if (element_type == GgufType::Array) {
        set_error(error, "nested GGUF arrays are not supported");
        return false;
    }
    for (std::uint64_t i = 0; i < count; ++i) {
        if (!skip_value(input, element_type, error)) {
            return false;
        }
    }
    return true;
}

bool skip_value(std::istream& input, GgufType type, std::string* error) {
    switch (type) {
    case GgufType::UInt8:
    case GgufType::Int8:
    case GgufType::Bool:
        return skip_bytes(input, 1, error);
    case GgufType::UInt16:
    case GgufType::Int16:
        return skip_bytes(input, 2, error);
    case GgufType::UInt32:
    case GgufType::Int32:
    case GgufType::Float32:
        return skip_bytes(input, 4, error);
    case GgufType::UInt64:
    case GgufType::Int64:
    case GgufType::Float64:
        return skip_bytes(input, 8, error);
    case GgufType::String: {
        std::string ignored;
        return read_string(input, &ignored, error);
    }
    case GgufType::Array:
        return skip_array(input, error);
    }
    set_error(error, "unsupported GGUF metadata type");
    return false;
}

bool read_metadata_value(std::istream& input,
                         GgufType type,
                         MetadataValue* value,
                         bool* supported,
                         std::string* error) {
    *supported = true;
    switch (type) {
    case GgufType::UInt8: {
        std::uint8_t number = 0;
        if (!read_pod(input, &number)) return false;
        *value = static_cast<std::uint64_t>(number);
        return true;
    }
    case GgufType::UInt16: {
        std::uint16_t number = 0;
        if (!read_pod(input, &number)) return false;
        *value = static_cast<std::uint64_t>(number);
        return true;
    }
    case GgufType::UInt32: {
        std::uint32_t number = 0;
        if (!read_pod(input, &number)) return false;
        *value = static_cast<std::uint64_t>(number);
        return true;
    }
    case GgufType::UInt64: {
        std::uint64_t number = 0;
        if (!read_pod(input, &number)) return false;
        *value = number;
        return true;
    }
    case GgufType::Int8: {
        std::int8_t number = 0;
        if (!read_pod(input, &number)) return false;
        *value = static_cast<std::int64_t>(number);
        return true;
    }
    case GgufType::Int16: {
        std::int16_t number = 0;
        if (!read_pod(input, &number)) return false;
        *value = static_cast<std::int64_t>(number);
        return true;
    }
    case GgufType::Int32: {
        std::int32_t number = 0;
        if (!read_pod(input, &number)) return false;
        *value = static_cast<std::int64_t>(number);
        return true;
    }
    case GgufType::Int64: {
        std::int64_t number = 0;
        if (!read_pod(input, &number)) return false;
        *value = number;
        return true;
    }
    case GgufType::String: {
        std::string text;
        if (!read_string(input, &text, error)) return false;
        *value = std::move(text);
        return true;
    }
    case GgufType::Float32: {
        float number = 0.0f;
        if (!read_pod(input, &number)) return false;
        *value = number;
        return true;
    }
    case GgufType::Float64: {
        double number = 0.0;
        if (!read_pod(input, &number)) return false;
        *value = static_cast<float>(number);
        return true;
    }
    case GgufType::Bool:
        *supported = false;
        return skip_value(input, type, error);
    case GgufType::Array: {
        std::uint32_t element_type_raw = 0;
        std::uint64_t count = 0;
        if (!read_pod(input, &element_type_raw) || !read_pod(input, &count)) {
            set_error(error, "unexpected end of GGUF while reading array header");
            return false;
        }
        if (count > kMaxArrayElements) {
            set_error(error, "GGUF metadata array is too large");
            return false;
        }
        const auto element_type = static_cast<GgufType>(element_type_raw);
        if (element_type == GgufType::String) {
            std::vector<std::string> items;
            items.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i) {
                std::string item;
                if (!read_string(input, &item, error)) return false;
                items.push_back(std::move(item));
            }
            *value = std::move(items);
            return true;
        }
        if (element_type == GgufType::Int32) {
            std::vector<std::int32_t> items;
            items.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i) {
                std::int32_t item = 0;
                if (!read_pod(input, &item)) return false;
                items.push_back(item);
            }
            *value = std::move(items);
            return true;
        }
        if (element_type == GgufType::UInt32) {
            std::vector<std::uint32_t> items;
            items.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i) {
                std::uint32_t item = 0;
                if (!read_pod(input, &item)) return false;
                items.push_back(item);
            }
            *value = std::move(items);
            return true;
        }
        if (element_type == GgufType::Float32) {
            std::vector<float> items;
            items.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i) {
                float item = 0.0f;
                if (!read_pod(input, &item)) return false;
                items.push_back(item);
            }
            *value = std::move(items);
            return true;
        }
        *supported = false;
        for (std::uint64_t i = 0; i < count; ++i) {
            if (!skip_value(input, element_type, error)) return false;
        }
        return true;
    }
    }
    set_error(error, "unsupported GGUF metadata type");
    return false;
}

bool align_stream(std::istream& input, std::uint64_t alignment, std::string* error) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        set_error(error, "GGUF alignment must be a non-zero power of two");
        return false;
    }
    const auto position = input.tellg();
    if (position < 0) {
        set_error(error, "unable to determine GGUF data offset");
        return false;
    }
    const auto current = static_cast<std::uint64_t>(position);
    const auto aligned = (current + alignment - 1) & ~(alignment - 1);
    return skip_bytes(input, aligned - current, error);
}

}  // namespace

bool GgufReader::read_index(const std::string& path,
                            GgufModelIndex* index,
                            std::string* error) {
    if (index == nullptr) {
        set_error(error, "GGUF index output must not be null");
        return false;
    }
    if (error != nullptr) error->clear();

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        set_error(error, "unable to open GGUF file: " + path);
        return false;
    }

    std::array<char, 4> magic{};
    if (!input.read(magic.data(), static_cast<std::streamsize>(magic.size())) ||
        magic != std::array<char, 4>{'G', 'G', 'U', 'F'}) {
        set_error(error, "invalid GGUF magic");
        return false;
    }

    GgufModelIndex parsed;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    if (!read_pod(input, &parsed.version) || !read_pod(input, &tensor_count) ||
        !read_pod(input, &metadata_count)) {
        set_error(error, "truncated GGUF header");
        return false;
    }
    if (parsed.version < kGgufVersionMin || parsed.version > kGgufVersionMax) {
        set_error(error, "unsupported GGUF version");
        return false;
    }
    if (metadata_count > kMaxMetadataCount || tensor_count > kMaxTensorCount) {
        set_error(error, "GGUF header count exceeds teaching reader limit");
        return false;
    }

    std::uint64_t alignment = kDefaultAlignment;
    std::unordered_set<std::string> metadata_keys;
    metadata_keys.reserve(static_cast<std::size_t>(metadata_count));
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        std::string key;
        std::uint32_t type_raw = 0;
        if (!read_string(input, &key, error) || !read_pod(input, &type_raw)) {
            set_error(error, "truncated GGUF metadata entry");
            return false;
        }
        MetadataValue value;
        bool supported = false;
        if (!read_metadata_value(input, static_cast<GgufType>(type_raw), &value, &supported, error)) {
            if (error != nullptr && error->empty()) {
                *error = "truncated GGUF metadata value";
            }
            return false;
        }
        if (key.empty()) {
            set_error(error, "GGUF metadata key must not be empty");
            return false;
        }
        if (!metadata_keys.emplace(key).second) {
            set_error(error, "duplicate GGUF metadata key: " + key);
            return false;
        }
        if (supported) {
            parsed.metadata.emplace(key, value);
            if (key == "general.alignment") {
                if (const auto* number = std::get_if<std::uint64_t>(&value)) {
                    alignment = *number;
                }
            }
        }
    }

    parsed.tensors.reserve(static_cast<std::size_t>(tensor_count));
    std::unordered_set<std::string> tensor_names;
    tensor_names.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensorInfo tensor;
        std::uint32_t dimension_count = 0;
        if (!read_string(input, &tensor.name, error) ||
            !read_pod(input, &dimension_count)) {
            set_error(error, "truncated GGUF tensor descriptor");
            return false;
        }
        if (tensor.name.empty()) {
            set_error(error, "GGUF tensor name must not be empty");
            return false;
        }
        if (!tensor_names.emplace(tensor.name).second) {
            set_error(error, "duplicate GGUF tensor name: " + tensor.name);
            return false;
        }
        if (dimension_count == 0 || dimension_count > kMaxTensorRank) {
            set_error(error, "GGUF tensor rank exceeds teaching reader limit");
            return false;
        }
        tensor.dimensions.resize(dimension_count);
        for (auto& dimension : tensor.dimensions) {
            if (!read_pod(input, &dimension)) {
                set_error(error, "truncated GGUF tensor dimensions");
                return false;
            }
            if (dimension == 0) {
                set_error(error, "GGUF tensor dimensions must be positive");
                return false;
            }
        }
        if (!read_pod(input, &tensor.type) || !read_pod(input, &tensor.offset)) {
            set_error(error, "truncated GGUF tensor descriptor");
            return false;
        }
        parsed.tensors.push_back(std::move(tensor));
    }

    if (!align_stream(input, alignment, error)) {
        return false;
    }
    const auto data_position = input.tellg();
    if (data_position < 0) {
        set_error(error, "unable to determine GGUF tensor data offset");
        return false;
    }
    parsed.data_offset = static_cast<std::uint64_t>(data_position);
    *index = std::move(parsed);
    if (error != nullptr) {
        error->clear();
    }
    return true;
}

}  // namespace chapter01
