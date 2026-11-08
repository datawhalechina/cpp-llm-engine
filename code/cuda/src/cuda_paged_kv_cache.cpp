// CUDA 执行层：实现共享分页 K/V payload 的物理槽位读写。
// 布局：[layer, physical_block, slot, kv_head, head_dimension]。
// 失败语义：参数在 CUDA 调用前完成校验，控制面提交由 Chapter07 单独负责。
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"

#include "cpp_llm_engine/cuda/cuda_status.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

namespace cpp_llm_engine::cuda {
namespace {

bool checked_payload_bytes(
    std::size_t layers,
    std::size_t blocks,
    std::size_t slots,
    std::size_t heads,
    std::size_t dimensions,
    std::size_t* output) {
    if (layers == 0 || blocks == 0 || slots == 0 || heads == 0 ||
        dimensions == 0 || output == nullptr) {
        return false;
    }
    std::size_t elements = 1;
    for (const std::size_t factor : {layers, blocks, slots, heads, dimensions}) {
        if (elements > std::numeric_limits<std::size_t>::max() / factor) return false;
        elements *= factor;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) return false;
    *output = elements * sizeof(float);
    return true;
}

bool matches_token(
    const CudaTensor& tensor,
    std::size_t heads,
    std::size_t dimensions) {
    const auto& contract = tensor.contract();
    return contract.dtype == core::DType::Float32 &&
        contract.device == core::DeviceKind::CUDA &&
        contract.shape.dimensions == std::vector<std::size_t>{1, heads, dimensions} &&
        contract.strides == std::vector<std::size_t>{heads * dimensions, dimensions, 1} &&
        contract.storage_offset_elements == 0;
}

}  // namespace

core::Status CudaPagedKvCache::create(
    std::size_t layer_count,
    std::size_t total_blocks,
    std::size_t block_size,
    std::size_t kv_head_count,
    std::size_t head_dimension,
    CudaPagedKvCache* output) {
    if (output == nullptr) {
        return core::Status::error("CUDA paged KV output is null");
    }
    std::size_t bytes = 0;
    if (!checked_payload_bytes(
            layer_count, total_blocks, block_size, kv_head_count,
            head_dimension, &bytes)) {
        return core::Status::error("CUDA paged KV dimensions overflow or are zero");
    }
    CudaPagedKvCache replacement;
    CudaStatus status = DeviceBuffer::allocate(bytes, &replacement.key_);
    if (!status.is_ok()) return status.to_core_status();
    status = DeviceBuffer::allocate(bytes, &replacement.value_);
    if (!status.is_ok()) return status.to_core_status();
    replacement.layer_count_ = layer_count;
    replacement.total_blocks_ = total_blocks;
    replacement.block_size_ = block_size;
    replacement.kv_head_count_ = kv_head_count;
    replacement.head_dimension_ = head_dimension;
    replacement.one_payload_bytes_ = bytes;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaPagedKvCache::write_token(
    std::size_t layer,
    std::size_t physical_block,
    std::size_t slot,
    const CudaTensor& key,
    const CudaTensor& value,
    const CudaStream& stream) {
    std::size_t offset = 0;
    if (!valid() || !stream.valid() ||
        !locate(layer, physical_block, slot, &offset) ||
        !matches_token(key, kv_head_count_, head_dimension_) ||
        !matches_token(value, kv_head_count_, head_dimension_)) {
        return core::Status::error("CUDA paged KV write input is invalid");
    }
    const std::size_t bytes = token_width() * sizeof(float);
    auto* key_destination = static_cast<unsigned char*>(key_.data()) + offset;
    auto* value_destination = static_cast<unsigned char*>(value_.data()) + offset;
    CudaStatus status = CudaStatus::from(
        cudaMemcpyAsync(
            key_destination, key.data(), bytes, cudaMemcpyDeviceToDevice,
            stream.native_handle()),
        "CUDA paged KV key D2D");
    if (!status.is_ok()) return status.to_core_status();
    return CudaStatus::from(
        cudaMemcpyAsync(
            value_destination, value.data(), bytes, cudaMemcpyDeviceToDevice,
            stream.native_handle()),
        "CUDA paged KV value D2D").to_core_status();
}

core::Status CudaPagedKvCache::copy_token_to_host_async(
    std::size_t layer,
    std::size_t physical_block,
    std::size_t slot,
    float* key,
    float* value,
    const CudaStream& stream) const {
    std::size_t offset = 0;
    if (!valid() || !stream.valid() || key == nullptr || value == nullptr ||
        !locate(layer, physical_block, slot, &offset)) {
        return core::Status::error("CUDA paged KV read input is invalid");
    }
    const std::size_t bytes = token_width() * sizeof(float);
    CudaStatus status = key_.copy_to_host_async(key, bytes, stream, offset);
    if (!status.is_ok()) return status.to_core_status();
    return value_.copy_to_host_async(value, bytes, stream, offset).to_core_status();
}

core::Status CudaPagedKvCache::clear_block(
    std::size_t physical_block,
    const CudaStream& stream) {
    if (!valid() || !stream.valid() || physical_block >= total_blocks_) {
        return core::Status::error("CUDA paged KV clear block input is invalid");
    }
    const std::size_t layer_bytes =
        block_size_ * token_width() * sizeof(float);
    for (std::size_t layer = 0; layer < layer_count_; ++layer) {
        const std::size_t offset =
            (layer * total_blocks_ + physical_block) * layer_bytes;
        auto status = CudaStatus::from(
            cudaMemsetAsync(
                static_cast<unsigned char*>(key_.data()) + offset,
                0,
                layer_bytes,
                stream.native_handle()),
            "CUDA paged KV key clear");
        if (!status.is_ok()) return status.to_core_status();
        status = CudaStatus::from(
            cudaMemsetAsync(
                static_cast<unsigned char*>(value_.data()) + offset,
                0,
                layer_bytes,
                stream.native_handle()),
            "CUDA paged KV value clear");
        if (!status.is_ok()) return status.to_core_status();
    }
    return core::Status::ok();
}

core::Status CudaPagedKvCache::reset() {
    const CudaStatus key_status = key_.reset();
    const CudaStatus value_status = value_.reset();
    layer_count_ = 0;
    total_blocks_ = 0;
    block_size_ = 0;
    kv_head_count_ = 0;
    head_dimension_ = 0;
    one_payload_bytes_ = 0;
    if (!key_status.is_ok()) return key_status.to_core_status();
    return value_status.to_core_status();
}

const float* CudaPagedKvCache::key_data() const {
    return static_cast<const float*>(key_.data());
}
const float* CudaPagedKvCache::value_data() const {
    return static_cast<const float*>(value_.data());
}
std::size_t CudaPagedKvCache::layer_count() const { return layer_count_; }
std::size_t CudaPagedKvCache::total_blocks() const { return total_blocks_; }
std::size_t CudaPagedKvCache::block_size() const { return block_size_; }
std::size_t CudaPagedKvCache::kv_head_count() const { return kv_head_count_; }
std::size_t CudaPagedKvCache::head_dimension() const { return head_dimension_; }
std::size_t CudaPagedKvCache::token_width() const {
    return kv_head_count_ * head_dimension_;
}
std::size_t CudaPagedKvCache::payload_bytes() const {
    return one_payload_bytes_ * 2;
}

bool CudaPagedKvCache::valid() const {
    std::size_t expected = 0;
    return checked_payload_bytes(
               layer_count_, total_blocks_, block_size_, kv_head_count_,
               head_dimension_, &expected) &&
        one_payload_bytes_ == expected && key_.size_bytes() == expected &&
        value_.size_bytes() == expected;
}

bool CudaPagedKvCache::locate(
    std::size_t layer,
    std::size_t physical_block,
    std::size_t slot,
    std::size_t* byte_offset) const {
    if (byte_offset == nullptr || layer >= layer_count_ ||
        physical_block >= total_blocks_ || slot >= block_size_) {
        return false;
    }
    const std::size_t token =
        (layer * total_blocks_ + physical_block) * block_size_ + slot;
    *byte_offset = token * token_width() * sizeof(float);
    return true;
}

}  // namespace cpp_llm_engine::cuda
