// CUDA 执行层：实现请求级连续 device KV Cache。
// 职责：校验 rank-3 K/V Tensor，并把新 token 复制到连续缓存尾部。
// 契约：失败不推进 token 数；clear 不清零 storage，后续 append 可覆盖旧数据。
// 验证：cpp_llm_engine_cuda_kv_cache_test 在真实 GPU 上执行。
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"

#include "cpp_llm_engine/core/tensor_contract.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <utility>

namespace cpp_llm_engine::cuda {
namespace {

bool checked_storage_bytes(
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dimension,
    std::size_t* output) {
    if (tokens == 0 || heads == 0 || head_dimension == 0 || output == nullptr) {
        return false;
    }
    constexpr std::size_t kElementBytes = sizeof(float);
    if (heads > std::numeric_limits<std::size_t>::max() / head_dimension) return false;
    const std::size_t width = heads * head_dimension;
    if (tokens > std::numeric_limits<std::size_t>::max() / width) return false;
    const std::size_t elements = tokens * width;
    if (elements > std::numeric_limits<std::size_t>::max() / kElementBytes) return false;
    *output = elements * kElementBytes;
    return true;
}

bool matches_input(
    const CudaTensor& tensor,
    std::size_t heads,
    std::size_t head_dimension,
    std::size_t* tokens) {
    const auto& contract = tensor.contract();
    if (contract.dtype != core::DType::Float32 ||
        contract.device != core::DeviceKind::CUDA ||
        contract.shape.dimensions.size() != 3 || contract.strides.size() != 3 ||
        contract.shape.dimensions[0] == 0 ||
        contract.shape.dimensions[1] != heads ||
        contract.shape.dimensions[2] != head_dimension ||
        contract.strides[0] != heads * head_dimension ||
        contract.strides[1] != head_dimension || contract.strides[2] != 1 ||
        contract.storage_offset_elements != 0) {
        return false;
    }
    *tokens = contract.shape.dimensions[0];
    return true;
}

}  // namespace

core::Status CudaKvCache::create(
    std::size_t capacity_tokens,
    std::size_t kv_head_count,
    std::size_t head_dimension,
    CudaKvCache* output) {
    if (output == nullptr) return core::Status::error("CudaKvCache output is null");
    std::size_t bytes = 0;
    if (!checked_storage_bytes(
            capacity_tokens, kv_head_count, head_dimension, &bytes)) {
        return core::Status::error("CudaKvCache dimensions overflow or are zero");
    }

    CudaKvCache replacement;
    CudaStatus status = DeviceBuffer::allocate(bytes, &replacement.key_);
    if (!status.is_ok()) return status.to_core_status();
    status = DeviceBuffer::allocate(bytes, &replacement.value_);
    if (!status.is_ok()) return status.to_core_status();
    replacement.capacity_tokens_ = capacity_tokens;
    replacement.kv_head_count_ = kv_head_count;
    replacement.head_dimension_ = head_dimension;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status CudaKvCache::append(
    const CudaTensor& key,
    const CudaTensor& value,
    const CudaStream& stream) {
    if (!valid()) return core::Status::error("CudaKvCache is not initialized");
    if (!stream.valid()) return core::Status::error("CudaKvCache stream is invalid");
    std::size_t key_tokens = 0;
    std::size_t value_tokens = 0;
    if (!matches_input(key, kv_head_count_, head_dimension_, &key_tokens) ||
        !matches_input(value, kv_head_count_, head_dimension_, &value_tokens) ||
        key_tokens != value_tokens) {
        return core::Status::error("CudaKvCache K/V shape mismatch");
    }
    if (key_tokens > capacity_tokens_ - size_tokens_) {
        return core::Status::error("CudaKvCache capacity exceeded");
    }

    const std::size_t offset_bytes =
        size_tokens_ * token_width() * sizeof(float);
    const std::size_t copy_bytes = key_tokens * token_width() * sizeof(float);
    auto* key_destination = static_cast<unsigned char*>(key_.data()) + offset_bytes;
    auto* value_destination = static_cast<unsigned char*>(value_.data()) + offset_bytes;
    CudaStatus status = CudaStatus::from(
        cudaMemcpyAsync(
            key_destination, key.data(), copy_bytes, cudaMemcpyDeviceToDevice,
            stream.native_handle()),
        "CudaKvCache key D2D");
    if (!status.is_ok()) return status.to_core_status();
    status = CudaStatus::from(
        cudaMemcpyAsync(
            value_destination, value.data(), copy_bytes, cudaMemcpyDeviceToDevice,
            stream.native_handle()),
        "CudaKvCache value D2D");
    if (!status.is_ok()) return status.to_core_status();
    size_tokens_ += key_tokens;
    return core::Status::ok();
}

core::Status CudaKvCache::copy_to_host_async(
    float* key,
    float* value,
    std::size_t token_count,
    const CudaStream& stream) const {
    if (!valid()) return core::Status::error("CudaKvCache is not initialized");
    if (token_count > size_tokens_) {
        return core::Status::error("CudaKvCache read exceeds stored tokens");
    }
    if (token_count == 0) return core::Status::ok();
    if (key == nullptr || value == nullptr) {
        return core::Status::error("CudaKvCache host output is null");
    }
    const std::size_t bytes = token_count * token_width() * sizeof(float);
    CudaStatus status = key_.copy_to_host_async(key, bytes, stream);
    if (!status.is_ok()) return status.to_core_status();
    return value_.copy_to_host_async(value, bytes, stream).to_core_status();
}

core::Status CudaKvCache::trim(std::size_t token_count) {
    if (!valid()) return core::Status::error("CudaKvCache is not initialized");
    if (token_count > size_tokens_) {
        return core::Status::error("CudaKvCache trim cannot grow the cache");
    }
    size_tokens_ = token_count;
    return core::Status::ok();
}

void CudaKvCache::clear() { size_tokens_ = 0; }

core::Status CudaKvCache::reset() {
    const CudaStatus key_status = key_.reset();
    const CudaStatus value_status = value_.reset();
    size_tokens_ = 0;
    capacity_tokens_ = 0;
    kv_head_count_ = 0;
    head_dimension_ = 0;
    if (!key_status.is_ok()) return key_status.to_core_status();
    return value_status.to_core_status();
}

const float* CudaKvCache::key_data() const {
    return static_cast<const float*>(key_.data());
}

const float* CudaKvCache::value_data() const {
    return static_cast<const float*>(value_.data());
}

std::size_t CudaKvCache::size_tokens() const { return size_tokens_; }
std::size_t CudaKvCache::capacity_tokens() const { return capacity_tokens_; }
std::size_t CudaKvCache::kv_head_count() const { return kv_head_count_; }
std::size_t CudaKvCache::head_dimension() const { return head_dimension_; }
std::size_t CudaKvCache::token_width() const {
    return kv_head_count_ * head_dimension_;
}

bool CudaKvCache::valid() const {
    std::size_t expected_bytes = 0;
    return checked_storage_bytes(
               capacity_tokens_, kv_head_count_, head_dimension_, &expected_bytes) &&
           key_.size_bytes() == expected_bytes && value_.size_bytes() == expected_bytes &&
           size_tokens_ <= capacity_tokens_;
}

}  // namespace cpp_llm_engine::cuda
