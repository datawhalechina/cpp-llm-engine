// Chapter 01：教学二进制模型 loader 实现
// 职责：读取教程自定义 CIE1 文件的 tensor records 和 payload。
// 契约：header、dtype、shape、byte size 和 payload 截断都必须校验。
// 验证：chapter01_tensor_loader_test.cpp。
#include "model_loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace chapter01 {

namespace {

constexpr char kMagic[4] = {'C', 'I', 'E', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kMaxTensorCount = 1U << 20U;
constexpr std::uint32_t kMaxTensorNameBytes = 1U << 20U;
constexpr std::uint32_t kMaxTensorRank = 8;

template <typename T>
bool read_pod(std::istream& in, T* value) {
    in.read(reinterpret_cast<char*>(value), sizeof(T));
    return static_cast<bool>(in);
}

bool read_string(std::istream& in, std::string* value) {
    std::uint32_t size = 0;
    if (!read_pod(in, &size)) {
        return false;
    }
    if (size == 0 || size > kMaxTensorNameBytes) return false;
    value->resize(size);
    if (size == 0) {
        return true;
    }
    in.read(value->data(), static_cast<std::streamsize>(size));
    return static_cast<bool>(in);
}

bool read_shape(std::istream& in, std::vector<std::int64_t>* shape) {
    std::uint32_t ndim = 0;
    if (!read_pod(in, &ndim)) {
        return false;
    }
    if (ndim == 0 || ndim > kMaxTensorRank) return false;
    shape->resize(ndim);
    for (std::uint32_t i = 0; i < ndim; ++i) {
        if (!read_pod(in, &(*shape)[i])) {
            return false;
        }
    }
    return true;
}

std::string make_error(const std::string& message) {
    return message;
}

bool checked_element_count(const std::vector<std::int64_t>& shape,
                           std::size_t* count) {
    std::size_t result = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0) return false;
        const auto converted = static_cast<std::size_t>(dimension);
        if (result > std::numeric_limits<std::size_t>::max() / converted) return false;
        result *= converted;
    }
    *count = result;
    return true;
}

}  // namespace

bool ModelLoader::load(const std::filesystem::path& path, std::string* error) {
    tensors_.clear();
    if (error != nullptr) error->clear();
    std::unordered_map<std::string, Tensor> loaded_tensors;

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        if (error) {
            *error = "failed to open model file: " + path.string();
        }
        return false;
    }

    char magic[4] = {};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        if (error) {
            *error = make_error("invalid model magic");
        }
        return false;
    }

    std::uint32_t version = 0;
    if (!read_pod(input, &version) || version != kVersion) {
        if (error) {
            *error = make_error("unsupported model version");
        }
        return false;
    }

    std::uint32_t tensor_count = 0;
    if (!read_pod(input, &tensor_count)) {
        if (error) {
            *error = make_error("failed to read tensor count");
        }
        return false;
    }
    if (tensor_count > kMaxTensorCount) {
        if (error) *error = make_error("tensor count exceeds teaching format limit");
        return false;
    }

    for (std::uint32_t i = 0; i < tensor_count; ++i) {
        std::string name;
        if (!read_string(input, &name)) {
            if (error) {
                *error = make_error("failed to read tensor name");
            }
            return false;
        }

        std::uint8_t dtype_u8 = 0;
        std::uint8_t device_u8 = 0;
        if (!read_pod(input, &dtype_u8) || !read_pod(input, &device_u8)) {
            if (error) {
                *error = make_error("failed to read tensor dtype/device");
            }
            return false;
        }
        const auto dtype = static_cast<DType>(dtype_u8);
        const auto device = static_cast<Device>(device_u8);
        const auto element_bytes = dtype_size(dtype);
        if (element_bytes == 0) {
            if (error) *error = make_error("unsupported teaching tensor dtype");
            return false;
        }
        if (device != Device::CPU) {
            if (error) *error = make_error("unsupported teaching tensor device");
            return false;
        }

        std::vector<std::int64_t> shape;
        if (!read_shape(input, &shape)) {
            if (error) {
                *error = make_error("failed to read tensor shape");
            }
            return false;
        }
        std::size_t element_count = 0;
        if (!checked_element_count(shape, &element_count) ||
            element_count > std::numeric_limits<std::size_t>::max() / element_bytes) {
            if (error) *error = make_error("invalid teaching tensor shape");
            return false;
        }
        const auto expected_bytes = element_count * element_bytes;

        std::uint64_t data_bytes_u64 = 0;
        if (!read_pod(input, &data_bytes_u64)) {
            if (error) {
                *error = make_error("failed to read tensor byte size");
            }
            return false;
        }
        if (data_bytes_u64 > std::numeric_limits<std::size_t>::max()) {
            if (error) {
                *error = make_error("tensor is too large");
            }
            return false;
        }
        if (data_bytes_u64 != expected_bytes) {
            if (error) *error = make_error("teaching tensor payload size does not match shape and dtype");
            return false;
        }
        if (data_bytes_u64 > static_cast<std::uint64_t>(
                                 std::numeric_limits<std::streamsize>::max())) {
            if (error) *error = make_error("tensor payload is not readable by this host");
            return false;
        }

        std::vector<std::byte> bytes(static_cast<std::size_t>(data_bytes_u64));
        if (!bytes.empty()) {
            input.read(reinterpret_cast<char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes.size()));
            if (!input) {
                if (error) {
                    *error = make_error("failed to read tensor payload");
                }
                return false;
            }
        }

        Tensor tensor = Tensor::from_raw_bytes(name,
                                               std::move(shape),
                                               dtype,
                                               std::move(bytes),
                                               device);
        if (!loaded_tensors.emplace(tensor.name(), std::move(tensor)).second) {
            if (error) *error = make_error("duplicate teaching tensor name");
            return false;
        }
    }

    tensors_ = std::move(loaded_tensors);
    if (error != nullptr) error->clear();
    return true;
}

const std::unordered_map<std::string, Tensor>& ModelLoader::tensors() const { return tensors_; }

const Tensor* ModelLoader::find(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<std::string> ModelLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        names.push_back(name);
    }
    return names;
}

}  // namespace chapter01
