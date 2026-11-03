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

template <typename T>
bool read_pod(std::istream& in, T* value) {
    in.read(reinterpret_cast<char*>(value), sizeof(T));
    return static_cast<bool>(in);
}

template <typename T>
bool write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(out);
}

bool read_string(std::istream& in, std::string* value) {
    std::uint32_t size = 0;
    if (!read_pod(in, &size)) {
        return false;
    }
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

}  // namespace

bool ModelLoader::load(const std::filesystem::path& path, std::string* error) {
    tensors_.clear();
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

        std::vector<std::int64_t> shape;
        if (!read_shape(input, &shape)) {
            if (error) {
                *error = make_error("failed to read tensor shape");
            }
            return false;
        }

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
                                               static_cast<DType>(dtype_u8),
                                               std::move(bytes),
                                               static_cast<Device>(device_u8));
        loaded_tensors.emplace(tensor.name(), std::move(tensor));
    }

    tensors_ = std::move(loaded_tensors);
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
