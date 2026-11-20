// CUDA 执行层：实现分页 MLA 压缩载荷的物理槽位读写。
// 布局：[layer, physical_block, slot, latent_or_rope_dimension]。
// 失败语义：参数先校验，控制面提交由 Chapter07 单独负责。
#include "cpp_llm_engine/cuda/cuda_paged_mla_cache.h"

#include "cpp_llm_engine/cuda/cuda_status.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <utility>

namespace cpp_llm_engine::cuda {
namespace {

bool checked_payload_bytes(
    std::size_t layers,
    std::size_t blocks,
    std::size_t slots,
    std::size_t width,
    std::size_t* output) {
    if (layers == 0 || blocks == 0 || slots == 0 || width == 0 ||
        output == nullptr) {
        return false;
    }
    std::size_t elements = 1;
    for (const std::size_t factor : {layers, blocks, slots, width}) {
        if (elements > std::numeric_limits<std::size_t>::max() / factor) {
            return false;
        }
        elements *= factor;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
    }
    *output = elements * sizeof(float);
    return true;
}

bool is_vector(const CudaTensor& tensor, std::size_t width) {
    const auto& contract = tensor.contract();
    std::size_t elements = 0;
    return !tensor.empty() && contract.dtype == core::DType::Float32 &&
        contract.device == core::DeviceKind::CUDA &&
        contract.shape.checked_numel(&elements).is_ok() && elements == width &&
        contract.storage_offset_elements == 0;
}

bool is_matrix(
    const CudaTensor& tensor, std::size_t width, std::size_t* rows) {
    const auto& contract = tensor.contract();
    const auto& shape = contract.shape.dimensions;
    std::size_t elements = 0;
    if (rows == nullptr || tensor.empty() ||
        contract.dtype != core::DType::Float32 ||
        contract.device != core::DeviceKind::CUDA || shape.size() < 2 ||
        shape[0] == 0 ||
        !contract.shape.checked_numel(&elements).is_ok() ||
        elements / shape[0] != width ||
        contract.storage_offset_elements != 0) {
        return false;
    }
    *rows = shape[0];
    return true;
}

}  // namespace

core::Status CudaPagedMlaCache::create(
    std::size_t layer_count,
    std::size_t total_blocks,
    std::size_t block_size,
    std::size_t latent_dimension,
    std::size_t rope_dimension,
    CudaPagedMlaCache* output) {
    if (output == nullptr) {
        return core::Status::error("CUDA paged MLA output is null");
    }
    std::size_t latent_bytes = 0;
    std::size_t rope_bytes = 0;
    if (!checked_payload_bytes(
            layer_count, total_blocks, block_size, latent_dimension,
            &latent_bytes) ||
        !checked_payload_bytes(
            layer_count, total_blocks, block_size, rope_dimension,
            &rope_bytes)) {
        return core::Status::error(
            "CUDA paged MLA dimensions overflow or are zero");
    }

    CudaPagedMlaCache replacement;
    CudaStatus status = DeviceBuffer::allocate(
        latent_bytes, &replacement.latent_);
    if (!status.is_ok()) return status.to_core_status();
    status = DeviceBuffer::allocate(rope_bytes, &replacement.key_rope_);
    if (!status.is_ok()) return status.to_core_status();
    replacement.layer_count_ = layer_count;
    replacement.total_blocks_ = total_blocks;
    replacement.block_size_ = block_size;
    replacement.latent_dimension_ = latent_dimension;
    replacement.rope_dimension_ = rope_dimension;
    replacement.latent_bytes_ = latent_bytes;
    replacement.rope_bytes_ = rope_bytes;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaPagedMlaCache::write_token(
    std::size_t layer,
    std::size_t physical_block,
    std::size_t slot,
    const CudaTensor& latent,
    const CudaTensor& key_rope,
    const CudaStream& stream) {
    std::size_t latent_offset = 0;
    std::size_t rope_offset = 0;
    if (!valid() || !stream.valid() ||
        !locate(
            layer, physical_block, slot, latent_dimension_, &latent_offset) ||
        !locate(
            layer, physical_block, slot, rope_dimension_, &rope_offset) ||
        !is_vector(latent, latent_dimension_) ||
        !is_vector(key_rope, rope_dimension_)) {
        return core::Status::error("CUDA paged MLA write input is invalid");
    }

    CudaStatus status = CudaStatus::from(
        cudaMemcpyAsync(
            static_cast<unsigned char*>(latent_.data()) + latent_offset,
            latent.data(),
            latent_dimension_ * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream.native_handle()),
        "CUDA paged MLA latent D2D");
    if (!status.is_ok()) return status.to_core_status();
    return CudaStatus::from(
        cudaMemcpyAsync(
            static_cast<unsigned char*>(key_rope_.data()) + rope_offset,
            key_rope.data(),
            rope_dimension_ * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream.native_handle()),
        "CUDA paged MLA RoPE D2D").to_core_status();
}

core::Status CudaPagedMlaCache::write_tokens(
    std::size_t layer,
    const std::vector<std::size_t>& physical_blocks,
    std::size_t start_position,
    const CudaTensor& latent,
    const CudaTensor& key_rope,
    const CudaStream& stream) {
    std::size_t latent_rows = 0;
    std::size_t rope_rows = 0;
    if (!valid() || !stream.valid() || layer >= layer_count_ ||
        physical_blocks.empty() ||
        !is_matrix(latent, latent_dimension_, &latent_rows) ||
        !is_matrix(key_rope, rope_dimension_, &rope_rows) ||
        latent_rows != rope_rows ||
        start_position > std::numeric_limits<std::size_t>::max() - latent_rows) {
        return core::Status::error("CUDA paged MLA batch write is invalid");
    }
    const std::size_t end_position = start_position + latent_rows;
    if (physical_blocks.size() >
            std::numeric_limits<std::size_t>::max() / block_size_ ||
        end_position > physical_blocks.size() * block_size_) {
        return core::Status::error(
            "CUDA paged MLA batch write exceeds the page table");
    }
    for (const std::size_t block : physical_blocks) {
        if (block >= total_blocks_) {
            return core::Status::error(
                "CUDA paged MLA batch physical block is out of range");
        }
    }

    const auto* latent_source = static_cast<const float*>(latent.data());
    const auto* rope_source = static_cast<const float*>(key_rope.data());
    for (std::size_t row = 0; row < latent_rows; ++row) {
        const std::size_t position = start_position + row;
        const std::size_t block = physical_blocks[position / block_size_];
        const std::size_t slot = position % block_size_;
        std::size_t latent_offset = 0;
        std::size_t rope_offset = 0;
        if (!locate(layer, block, slot, latent_dimension_, &latent_offset) ||
            !locate(layer, block, slot, rope_dimension_, &rope_offset)) {
            return core::Status::error(
                "CUDA paged MLA batch destination is invalid");
        }
        CudaStatus status = CudaStatus::from(cudaMemcpyAsync(
            static_cast<unsigned char*>(latent_.data()) + latent_offset,
            latent_source + row * latent_dimension_,
            latent_dimension_ * sizeof(float), cudaMemcpyDeviceToDevice,
            stream.native_handle()), "CUDA paged MLA latent batch D2D");
        if (!status.is_ok()) return status.to_core_status();
        status = CudaStatus::from(cudaMemcpyAsync(
            static_cast<unsigned char*>(key_rope_.data()) + rope_offset,
            rope_source + row * rope_dimension_,
            rope_dimension_ * sizeof(float), cudaMemcpyDeviceToDevice,
            stream.native_handle()), "CUDA paged MLA RoPE batch D2D");
        if (!status.is_ok()) return status.to_core_status();
    }
    return core::Status::ok();
}

core::Status CudaPagedMlaCache::copy_token_to_host_async(
    std::size_t layer,
    std::size_t physical_block,
    std::size_t slot,
    float* latent,
    float* key_rope,
    const CudaStream& stream) const {
    std::size_t latent_offset = 0;
    std::size_t rope_offset = 0;
    if (!valid() || !stream.valid() || latent == nullptr || key_rope == nullptr ||
        !locate(
            layer, physical_block, slot, latent_dimension_, &latent_offset) ||
        !locate(
            layer, physical_block, slot, rope_dimension_, &rope_offset)) {
        return core::Status::error("CUDA paged MLA read input is invalid");
    }
    CudaStatus status = latent_.copy_to_host_async(
        latent,
        latent_dimension_ * sizeof(float),
        stream,
        latent_offset);
    if (!status.is_ok()) return status.to_core_status();
    return key_rope_.copy_to_host_async(
        key_rope,
        rope_dimension_ * sizeof(float),
        stream,
        rope_offset).to_core_status();
}

core::Status CudaPagedMlaCache::clear_block(
    std::size_t physical_block,
    const CudaStream& stream) {
    if (!valid() || !stream.valid() || physical_block >= total_blocks_) {
        return core::Status::error("CUDA paged MLA clear block input is invalid");
    }
    const std::size_t latent_layer_bytes =
        block_size_ * latent_dimension_ * sizeof(float);
    const std::size_t rope_layer_bytes =
        block_size_ * rope_dimension_ * sizeof(float);
    for (std::size_t layer = 0; layer < layer_count_; ++layer) {
        const std::size_t latent_offset =
            (layer * total_blocks_ + physical_block) * latent_layer_bytes;
        const std::size_t rope_offset =
            (layer * total_blocks_ + physical_block) * rope_layer_bytes;
        CudaStatus status = CudaStatus::from(
            cudaMemsetAsync(
                static_cast<unsigned char*>(latent_.data()) + latent_offset,
                0,
                latent_layer_bytes,
                stream.native_handle()),
            "CUDA paged MLA latent clear");
        if (!status.is_ok()) return status.to_core_status();
        status = CudaStatus::from(
            cudaMemsetAsync(
                static_cast<unsigned char*>(key_rope_.data()) + rope_offset,
                0,
                rope_layer_bytes,
                stream.native_handle()),
            "CUDA paged MLA RoPE clear");
        if (!status.is_ok()) return status.to_core_status();
    }
    return core::Status::ok();
}

core::Status CudaPagedMlaCache::reset() {
    const CudaStatus latent_status = latent_.reset();
    const CudaStatus rope_status = key_rope_.reset();
    layer_count_ = 0;
    total_blocks_ = 0;
    block_size_ = 0;
    latent_dimension_ = 0;
    rope_dimension_ = 0;
    latent_bytes_ = 0;
    rope_bytes_ = 0;
    if (!latent_status.is_ok()) return latent_status.to_core_status();
    return rope_status.to_core_status();
}

const float* CudaPagedMlaCache::latent_data() const {
    return static_cast<const float*>(latent_.data());
}
const float* CudaPagedMlaCache::key_rope_data() const {
    return static_cast<const float*>(key_rope_.data());
}
std::size_t CudaPagedMlaCache::layer_count() const { return layer_count_; }
std::size_t CudaPagedMlaCache::total_blocks() const { return total_blocks_; }
std::size_t CudaPagedMlaCache::block_size() const { return block_size_; }
std::size_t CudaPagedMlaCache::latent_dimension() const {
    return latent_dimension_;
}
std::size_t CudaPagedMlaCache::rope_dimension() const {
    return rope_dimension_;
}
std::size_t CudaPagedMlaCache::payload_bytes() const {
    return latent_bytes_ + rope_bytes_;
}

bool CudaPagedMlaCache::valid() const {
    std::size_t expected_latent = 0;
    std::size_t expected_rope = 0;
    return checked_payload_bytes(
               layer_count_, total_blocks_, block_size_, latent_dimension_,
               &expected_latent) &&
        checked_payload_bytes(
               layer_count_, total_blocks_, block_size_, rope_dimension_,
               &expected_rope) &&
        latent_bytes_ == expected_latent && rope_bytes_ == expected_rope &&
        latent_.size_bytes() == expected_latent &&
        key_rope_.size_bytes() == expected_rope;
}

bool CudaPagedMlaCache::locate(
    std::size_t layer,
    std::size_t physical_block,
    std::size_t slot,
    std::size_t width,
    std::size_t* byte_offset) const {
    if (byte_offset == nullptr || layer >= layer_count_ ||
        physical_block >= total_blocks_ || slot >= block_size_) {
        return false;
    }
    const std::size_t token =
        (layer * total_blocks_ + physical_block) * block_size_ + slot;
    *byte_offset = token * width * sizeof(float);
    return true;
}

}  // namespace cpp_llm_engine::cuda
