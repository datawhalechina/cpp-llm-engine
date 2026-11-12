// CUDA 执行层：实现请求级压缩 MLA Cache 的分配、追加和回退。
#include "cpp_llm_engine/cuda/cuda_mla_cache.h"

#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <utility>
#include <vector>

namespace cpp_llm_engine::cuda {
namespace {

bool checked_bytes(
    std::size_t tokens, std::size_t width, std::size_t* output) {
    if (tokens == 0 || width == 0 || output == nullptr ||
        tokens > std::numeric_limits<std::size_t>::max() / width ||
        tokens * width > std::numeric_limits<std::size_t>::max() /
                             sizeof(float)) {
        return false;
    }
    *output = tokens * width * sizeof(float);
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

}  // namespace

core::Status CudaMlaCache::create(
    std::size_t capacity_tokens,
    std::size_t latent_dimension,
    std::size_t rope_dimension,
    CudaMlaCache* output) {
    if (output == nullptr) return core::Status::error("MLA Cache output is null");
    std::size_t latent_bytes = 0;
    std::size_t rope_bytes = 0;
    if (!checked_bytes(capacity_tokens, latent_dimension, &latent_bytes) ||
        !checked_bytes(capacity_tokens, rope_dimension, &rope_bytes)) {
        return core::Status::error("MLA Cache dimensions overflow or are zero");
    }
    CudaMlaCache replacement;
    CudaStatus status = DeviceBuffer::allocate(latent_bytes, &replacement.latent_);
    if (!status.is_ok()) return status.to_core_status();
    status = DeviceBuffer::allocate(rope_bytes, &replacement.key_rope_);
    if (!status.is_ok()) return status.to_core_status();
    replacement.capacity_tokens_ = capacity_tokens;
    replacement.latent_dimension_ = latent_dimension;
    replacement.rope_dimension_ = rope_dimension;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaMlaCache::append(
    const CudaTensor& latent,
    const CudaTensor& key_rope,
    const CudaStream& stream) {
    if (!valid() || !stream.valid()) {
        return core::Status::error("MLA Cache or stream is invalid");
    }
    if (!is_vector(latent, latent_dimension_) ||
        !is_vector(key_rope, rope_dimension_)) {
        return core::Status::error("MLA Cache append shape mismatch");
    }
    if (size_tokens_ == capacity_tokens_) {
        return core::Status::error("MLA Cache capacity exceeded");
    }
    auto* latent_destination = static_cast<float*>(latent_.data()) +
        size_tokens_ * latent_dimension_;
    auto* rope_destination = static_cast<float*>(key_rope_.data()) +
        size_tokens_ * rope_dimension_;
    CudaStatus status = CudaStatus::from(cudaMemcpyAsync(
        latent_destination, latent.data(), latent.size_bytes(),
        cudaMemcpyDeviceToDevice, stream.native_handle()), "MLA latent append");
    if (!status.is_ok()) return status.to_core_status();
    status = CudaStatus::from(cudaMemcpyAsync(
        rope_destination, key_rope.data(), key_rope.size_bytes(),
        cudaMemcpyDeviceToDevice, stream.native_handle()), "MLA rope append");
    if (!status.is_ok()) return status.to_core_status();
    ++size_tokens_;
    return core::Status::ok();
}

core::Status CudaMlaCache::copy_to_host_async(
    float* latent,
    float* key_rope,
    std::size_t token_count,
    const CudaStream& stream) const {
    if (!valid() || !stream.valid() || token_count > size_tokens_) {
        return core::Status::error("MLA Cache read is invalid");
    }
    if (token_count == 0) return core::Status::ok();
    if (latent == nullptr || key_rope == nullptr) {
        return core::Status::error("MLA Cache host output is null");
    }
    CudaStatus status = latent_.copy_to_host_async(
        latent, token_count * latent_dimension_ * sizeof(float), stream);
    if (!status.is_ok()) return status.to_core_status();
    return key_rope_.copy_to_host_async(
        key_rope, token_count * rope_dimension_ * sizeof(float), stream)
        .to_core_status();
}

core::Status CudaMlaCache::trim(std::size_t token_count) {
    if (!valid() || token_count > size_tokens_) {
        return core::Status::error("MLA Cache trim cannot grow the cache");
    }
    size_tokens_ = token_count;
    return core::Status::ok();
}

void CudaMlaCache::clear() { size_tokens_ = 0; }
const float* CudaMlaCache::latent_data() const {
    return static_cast<const float*>(latent_.data());
}
const float* CudaMlaCache::key_rope_data() const {
    return static_cast<const float*>(key_rope_.data());
}
std::size_t CudaMlaCache::size_tokens() const { return size_tokens_; }
std::size_t CudaMlaCache::capacity_tokens() const { return capacity_tokens_; }
std::size_t CudaMlaCache::latent_dimension() const { return latent_dimension_; }
std::size_t CudaMlaCache::rope_dimension() const { return rope_dimension_; }
bool CudaMlaCache::valid() const {
    std::size_t latent_bytes = 0;
    std::size_t rope_bytes = 0;
    return checked_bytes(capacity_tokens_, latent_dimension_, &latent_bytes) &&
        checked_bytes(capacity_tokens_, rope_dimension_, &rope_bytes) &&
        latent_.size_bytes() == latent_bytes &&
        key_rope_.size_bytes() == rope_bytes && size_tokens_ <= capacity_tokens_;
}

}  // namespace cpp_llm_engine::cuda
