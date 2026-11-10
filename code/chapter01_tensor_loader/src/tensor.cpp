// Chapter 01：loader Tensor 实现
// 职责：保存 dtype、shape、device 和 raw bytes，供 loader 与 schema 消费。
// 契约：对象不负责数值 kernel；类型化读写必须先核对 dtype、shape 与 storage。
// 验证：chapter01_tensor_loader_test.cpp。
#include "tensor.h"

#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace chapter01 {

namespace {

std::size_t product(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    std::size_t n = 1;
    for (auto dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("tensor shape cannot contain negative dimensions");
        }
        const auto dimension = static_cast<std::size_t>(dim);
        if (dimension != 0 && n > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor shape size overflows host size");
        }
        n *= dimension;
    }
    return n;
}

}  // namespace

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return sizeof(float);
        case DType::Int8:
            return sizeof(std::int8_t);
        case DType::Float16:
            return sizeof(std::uint16_t);
        case DType::UInt8:
            return sizeof(std::uint8_t);
        case DType::Unknown:
        default:
            return 0;
    }
}

Tensor::Tensor() : storage_(std::make_shared<Storage>()) {}

Tensor::Tensor(std::string name, std::vector<std::int64_t> shape, DType dtype, Device device)
    : name_(std::move(name)),
      shape_(std::move(shape)),
      dtype_(dtype),
      device_(device),
      storage_(std::make_shared<Storage>()) {}

Tensor Tensor::from_raw_bytes(std::string name,
                              std::vector<std::int64_t> shape,
                              DType dtype,
                              std::vector<std::byte> bytes,
                              Device device) {
    Tensor tensor(std::move(name), std::move(shape), dtype, device);
    tensor.payload_bytes_ = bytes.size();
    tensor.set_storage(std::make_shared<Storage>(std::move(bytes)), 0);
    return tensor;
}

const std::string& Tensor::name() const { return name_; }

void Tensor::set_name(std::string name) { name_ = std::move(name); }

const std::vector<std::int64_t>& Tensor::shape() const { return shape_; }

void Tensor::set_shape(std::vector<std::int64_t> shape) { shape_ = std::move(shape); }

DType Tensor::dtype() const { return dtype_; }

void Tensor::set_dtype(DType dtype) { dtype_ = dtype; }

Device Tensor::device() const { return device_; }

void Tensor::set_device(Device device) { device_ = device; }

std::size_t Tensor::numel() const { return product(shape_); }

std::size_t Tensor::bytes() const {
    if (dtype_ == DType::Q3_K || dtype_ == DType::Q4_K ||
        dtype_ == DType::Q5_0 || dtype_ == DType::Q5_1 ||
        dtype_ == DType::Q5_K || dtype_ == DType::Q6_K ||
        dtype_ == DType::Q8_0) {
        return payload_bytes_;
    }
    const std::size_t element_bytes = dtype_size(dtype_);
    if (element_bytes != 0 && numel() > std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::overflow_error("tensor byte size overflows host size");
    }
    return numel() * element_bytes;
}

const std::byte* Tensor::raw_data() const {
    if (!storage_) {
        return nullptr;
    }
    if (offset_bytes_ > storage_->size()) {
        return nullptr;
    }
    return storage_->data() + offset_bytes_;
}

std::byte* Tensor::raw_data() {
    if (!storage_) {
        return nullptr;
    }
    if (offset_bytes_ > storage_->size()) {
        return nullptr;
    }
    return storage_->data() + offset_bytes_;
}

const Tensor::StoragePtr& Tensor::storage() const { return storage_; }

void Tensor::set_storage(StoragePtr storage, std::size_t offset_bytes) {
    if (storage == nullptr || offset_bytes > storage->size()) {
        throw std::invalid_argument("tensor storage offset is outside storage");
    }
    storage_ = std::move(storage);
    offset_bytes_ = offset_bytes;
}

std::size_t Tensor::offset_bytes() const { return offset_bytes_; }

void Tensor::set_offset_bytes(std::size_t offset_bytes) {
    if (!storage_ || offset_bytes > storage_->size() ||
        bytes() > storage_->size() - offset_bytes) {
        throw std::invalid_argument("tensor storage offset leaves an incomplete payload");
    }
    offset_bytes_ = offset_bytes;
}

}  // namespace chapter01
