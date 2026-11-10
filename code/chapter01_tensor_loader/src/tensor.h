// Chapter 01：loader Tensor 对象
// 职责：表达从模型文件加载出的单个 tensor 元数据和 payload。
// 契约：支持 Float32、Float16、Int8 和教程使用的 GGUF 量化 dtype raw storage。
// 验证：chapter01_tensor_loader_test.cpp。
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace chapter01 {

enum class DType : std::uint8_t {
    Unknown = 0,
    Float32 = 1,
    Int8 = 2,
    Float16 = 3,
    UInt8 = 4,
    Q4_K = 5,
    Q6_K = 6,
    Q8_0 = 7,
    Q3_K = 8,
    Q5_1 = 9,
    Q5_K = 10,
    Q5_0 = 11,
    Q2_K = 12,
    IQ4_NL = 13,
};

enum class Device : std::uint8_t {
    CPU = 0,
};

std::size_t dtype_size(DType dtype);

class Tensor {
public:
    using Storage = std::vector<std::byte>;
    using StoragePtr = std::shared_ptr<Storage>;

    Tensor();
    Tensor(std::string name, std::vector<std::int64_t> shape, DType dtype, Device device = Device::CPU);

    static Tensor from_raw_bytes(std::string name,
                                 std::vector<std::int64_t> shape,
                                 DType dtype,
                                 std::vector<std::byte> bytes,
                                 Device device = Device::CPU);

    template <typename T>
    static Tensor from_vector(std::string name,
                              std::vector<std::int64_t> shape,
                              const std::vector<T>& values,
                              Device device = Device::CPU) {
        Tensor tensor(std::move(name), std::move(shape), deduce_dtype<T>(), device);
        tensor.set_vector(values);
        return tensor;
    }

    const std::string& name() const;
    void set_name(std::string name);

    const std::vector<std::int64_t>& shape() const;
    void set_shape(std::vector<std::int64_t> shape);

    DType dtype() const;
    void set_dtype(DType dtype);

    Device device() const;
    void set_device(Device device);

    std::size_t numel() const;
    std::size_t bytes() const;

    const std::byte* raw_data() const;
    std::byte* raw_data();

    const StoragePtr& storage() const;
    void set_storage(StoragePtr storage, std::size_t offset_bytes = 0);
    std::size_t offset_bytes() const;
    void set_offset_bytes(std::size_t offset_bytes);

    template <typename T>
    std::vector<T> to_vector() const {
        const auto expected_dtype = deduce_dtype<T>();
        if (expected_dtype == DType::Unknown || dtype_ != expected_dtype) {
            throw std::invalid_argument("tensor dtype does not match requested vector type");
        }
        const auto element_count = numel();
        if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("tensor vector byte size overflows host size");
        }
        const auto required_bytes = element_count * sizeof(T);
        if (!storage_ || offset_bytes_ > storage_->size() ||
            required_bytes > storage_->size() - offset_bytes_) {
            throw std::out_of_range("tensor storage is shorter than requested vector");
        }
        std::vector<T> values(element_count);
        if (values.empty()) {
            return values;
        }
        const auto* src = raw_data();
        std::memcpy(values.data(), src, values.size() * sizeof(T));
        return values;
    }

    template <typename T>
    void set_vector(const std::vector<T>& values) {
        const auto vector_dtype = deduce_dtype<T>();
        if (vector_dtype == DType::Unknown) {
            throw std::invalid_argument("tensor vector type is not supported");
        }
        if (numel() != values.size()) {
            throw std::invalid_argument("tensor vector size does not match shape");
        }
        set_dtype(vector_dtype);
        auto storage_bytes = values.size() * sizeof(T);
        auto storage = std::make_shared<Storage>(storage_bytes);
        if (storage_bytes > 0) {
            std::memcpy(storage->data(), values.data(), storage_bytes);
        }
        set_storage(std::move(storage), 0);
    }

    template <typename T>
    static constexpr DType deduce_dtype();

private:
    std::string name_;
    std::vector<std::int64_t> shape_;
    DType dtype_{DType::Unknown};
    Device device_{Device::CPU};
    StoragePtr storage_;
    std::size_t offset_bytes_{0};
    std::size_t payload_bytes_{0};
};

template <typename T>
constexpr DType Tensor::deduce_dtype() {
    if constexpr (std::is_same_v<T, float>) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return DType::Int8;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return DType::UInt8;
    } else {
        return DType::Unknown;
    }
}

}  // namespace chapter01
