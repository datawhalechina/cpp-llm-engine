// CUDA Backend：实现 Q4_K/Q6_K 解码与 GEMV 融合 kernel。
// 契约：一个 CUDA block 负责一个输出行，线程直接累加 packed 权重点积。
// 验证：cuda_quantized_linear_test.cu 与 Chapter05 CPU reference 对照。
#include "cpp_llm_engine/cuda/quantized_gemv_backend.h"

#include "cpp_llm_engine/core/quantization_contract.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr unsigned int kThreads = 256;

__device__ float half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::uint32_t normalized = mantissa;
            std::uint32_t adjusted_exponent = 127 - 14;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --adjusted_exponent;
            }
            normalized &= 0x03ffu;
            result = sign | (adjusted_exponent << 23) | (normalized << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    return __uint_as_float(result);
}

__device__ std::uint16_t load_u16(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) |
        (static_cast<std::uint16_t>(data[1]) << 8);
}

__device__ void q4k_scale_min(
    const std::uint8_t* scales,
    int index,
    std::uint8_t* scale,
    std::uint8_t* minimum) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 0x0f) |
            ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) |
            ((scales[index] >> 6) << 4);
    }
}

__device__ float decode_q4k(
    const std::uint8_t* block,
    std::size_t local) {
    const float d = half_bits_to_float(load_u16(block));
    const float dmin = half_bits_to_float(load_u16(block + 2));
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* quants = block + 16;
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    q4k_scale_min(scales, static_cast<int>(local / 32), &scale, &minimum);
    const std::size_t quant_index = (local / 64) * 32 + local % 32;
    const std::uint8_t packed = quants[quant_index];
    const std::uint8_t quant = local % 64 < 32 ? packed & 0x0f : packed >> 4;
    return d * static_cast<float>(scale) * static_cast<float>(quant) -
        dmin * static_cast<float>(minimum);
}

__device__ float decode_q6k(
    const std::uint8_t* block,
    std::size_t local) {
    const std::size_t half = local / 128;
    const std::size_t within_half = local % 128;
    const std::size_t segment = within_half / 32;
    const std::size_t lane = within_half % 32;
    const std::uint8_t* ql = block + half * 64;
    const std::uint8_t* qh = block + 128 + half * 32;
    const auto* scales = reinterpret_cast<const std::int8_t*>(
        block + 192 + half * 8);
    const std::uint8_t low =
        segment == 0 || segment == 2 ? ql[lane] : ql[lane + 32];
    const std::uint8_t nibble = segment < 2 ? low & 0x0f : low >> 4;
    const std::uint8_t high = (qh[lane] >> (segment * 2)) & 0x03;
    const int quant = static_cast<int>(nibble | (high << 4)) - 32;
    const int scale = scales[segment * 2 + lane / 16];
    const float d = half_bits_to_float(load_u16(block + 208));
    return d * static_cast<float>(scale * quant);
}

template <bool Q4K>
__global__ void quantized_gemv_kernel(
    const std::uint8_t* weights,
    const float* input,
    float* output,
    std::size_t cols) {
    const std::size_t row = blockIdx.x;
    const std::size_t blocks_per_row = cols / core::kQuantizedBlockElements;
    constexpr std::size_t block_bytes =
        Q4K ? core::kQ4KBlockBytes : core::kQ6KBlockBytes;
    const std::uint8_t* row_weights =
        weights + row * blocks_per_row * block_bytes;

    float sum = 0.0F;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* packed = row_weights + block * block_bytes;
        const std::size_t local = threadIdx.x;
        const float weight = Q4K
            ? decode_q4k(packed, local)
            : decode_q6k(packed, local);
        sum += weight * input[block * core::kQuantizedBlockElements + local];
    }

    __shared__ float partial[kThreads];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partial[0];
}

}  // namespace

core::BackendKind CudaQuantizedGemvBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaQuantizedGemvBackend::name() const {
    return "cuda-q4k-q6k-fused-gemv";
}

CudaStatus CudaQuantizedGemvBackend::compute(
    const CudaQuantizedMatrix& weights,
    const float* input,
    std::size_t input_elements,
    float* output,
    std::size_t output_elements,
    const CudaStream& stream) const {
    if (weights.empty() || input == nullptr || output == nullptr) {
        return CudaStatus::from(
            cudaErrorInvalidDevicePointer,
            "CudaQuantizedGemvBackend::compute pointers");
    }
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle,
            "CudaQuantizedGemvBackend::compute stream");
    }
    if (weights.cols() % core::kQuantizedBlockElements != 0 ||
        input_elements != weights.cols() || output_elements != weights.rows() ||
        weights.rows() > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedGemvBackend::compute shape");
    }

    const auto* packed = static_cast<const std::uint8_t*>(weights.data());
    const unsigned int grid = static_cast<unsigned int>(weights.rows());
    if (weights.format() == core::WeightFormat::Q4K) {
        quantized_gemv_kernel<true><<<grid, kThreads, 0, stream.native_handle()>>>(
            packed, input, output, weights.cols());
    } else if (weights.format() == core::WeightFormat::Q6K) {
        quantized_gemv_kernel<false><<<grid, kThreads, 0, stream.native_handle()>>>(
            packed, input, output, weights.cols());
    } else {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedGemvBackend::compute format");
    }
    return check_last_cuda_error("quantized fused GEMV kernel launch");
}

const CudaQuantizedGemvBackend& cuda_quantized_gemv_backend() {
    static const CudaQuantizedGemvBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
