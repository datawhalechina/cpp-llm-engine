// CUDA Backend：实现 GGML Q4_K/Q6_K block 到 Float32 workspace 的解码。
// 契约：每个线程解码一个权重值，保持 Chapter05 CPU reference 的元素顺序。
// 验证：cpp_llm_engine_cuda_quantized_linear_test 与 CPU reference 对照。
#include "cpp_llm_engine/cuda/quantized_dequantize_backend.h"

#include "cpp_llm_engine/core/linear_contract.h"
#include "cpp_llm_engine/core/quantization_contract.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr std::size_t kBlockSize = 256;

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

__global__ void dequantize_q4k_kernel(
    const std::uint8_t* input,
    float* output,
    std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const std::size_t block_index = index / core::kQuantizedBlockElements;
    const std::size_t local = index % core::kQuantizedBlockElements;
    const std::uint8_t* block = input + block_index * core::kQ4KBlockBytes;
    const float d = half_bits_to_float(load_u16(block));
    const float dmin = half_bits_to_float(load_u16(block + 2));
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* quants = block + 16;
    const int scale_index = static_cast<int>(local / 32);
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    q4k_scale_min(scales, scale_index, &scale, &minimum);
    const std::size_t quant_index = (local / 64) * 32 + local % 32;
    const std::uint8_t packed = quants[quant_index];
    const std::uint8_t quant = local % 64 < 32 ? packed & 0x0f : packed >> 4;
    output[index] = d * static_cast<float>(scale) * static_cast<float>(quant) -
        dmin * static_cast<float>(minimum);
}

__global__ void dequantize_q6k_kernel(
    const std::uint8_t* input,
    float* output,
    std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const std::size_t block_index = index / core::kQuantizedBlockElements;
    const std::size_t local = index % core::kQuantizedBlockElements;
    const std::uint8_t* block = input + block_index * core::kQ6KBlockBytes;
    const std::size_t half = local / 128;
    const std::size_t within_half = local % 128;
    const std::size_t segment = within_half / 32;
    const std::size_t lane = within_half % 32;
    const std::uint8_t* ql = block + half * 64;
    const std::uint8_t* qh = block + 128 + half * 32;
    const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192 + half * 8);
    const std::uint8_t low = segment == 0 || segment == 2 ? ql[lane] : ql[lane + 32];
    const std::uint8_t nibble = segment < 2 ? low & 0x0f : low >> 4;
    const std::uint8_t high = (qh[lane] >> (segment * 2)) & 0x03;
    const int quant = static_cast<int>(nibble | (high << 4)) - 32;
    const int scale = scales[segment * 2 + lane / 16];
    const float d = half_bits_to_float(load_u16(block + 208));
    output[index] = d * static_cast<float>(scale * quant);
}

}  // namespace

core::BackendKind CudaQuantizedDequantizeBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaQuantizedDequantizeBackend::name() const {
    return "cuda-q4k-q6k-dequantize";
}

CudaStatus CudaQuantizedDequantizeBackend::dequantize(
    const CudaQuantizedMatrix& weights,
    float* output,
    std::size_t output_elements,
    const CudaStream& stream) const {
    if (weights.empty() || output == nullptr) {
        return CudaStatus::from(
            cudaErrorInvalidDevicePointer,
            "CudaQuantizedDequantizeBackend::dequantize pointers");
    }
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle,
            "CudaQuantizedDequantizeBackend::dequantize stream");
    }
    if (weights.rows() > std::numeric_limits<std::size_t>::max() / weights.cols() ||
        output_elements != weights.rows() * weights.cols()) {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedDequantizeBackend::dequantize elements");
    }
    const std::size_t blocks = output_elements / kBlockSize +
        (output_elements % kBlockSize != 0 ? 1 : 0);
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidConfiguration,
            "CudaQuantizedDequantizeBackend::dequantize grid");
    }

    const auto* input = static_cast<const std::uint8_t*>(weights.data());
    if (weights.format() == core::WeightFormat::Q4K) {
        dequantize_q4k_kernel<<<
            static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(kBlockSize),
            0,
            stream.native_handle()>>>(input, output, output_elements);
    } else if (weights.format() == core::WeightFormat::Q6K) {
        dequantize_q6k_kernel<<<
            static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(kBlockSize),
            0,
            stream.native_handle()>>>(input, output, output_elements);
    } else {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedDequantizeBackend::dequantize format");
    }
    return check_last_cuda_error("quantized dequantize kernel launch");
}

CudaStatus CudaQuantizedDequantizeBackend::dequantize_row(
    const CudaQuantizedMatrix& weights,
    std::size_t row,
    float* output,
    std::size_t output_elements,
    const CudaStream& stream) const {
    if (weights.empty() || output == nullptr) {
        return CudaStatus::from(
            cudaErrorInvalidDevicePointer,
            "CudaQuantizedDequantizeBackend::dequantize_row pointers");
    }
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle,
            "CudaQuantizedDequantizeBackend::dequantize_row stream");
    }
    if (row >= weights.rows() || output_elements != weights.cols()) {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedDequantizeBackend::dequantize_row shape");
    }
    const std::size_t blocks = output_elements / kBlockSize +
        (output_elements % kBlockSize != 0 ? 1 : 0);
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidConfiguration,
            "CudaQuantizedDequantizeBackend::dequantize_row grid");
    }
    const std::size_t row_bytes = weights.size_bytes() / weights.rows();
    const auto* input = static_cast<const std::uint8_t*>(weights.data()) +
        row * row_bytes;
    if (weights.format() == core::WeightFormat::Q4K) {
        dequantize_q4k_kernel<<<
            static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(kBlockSize),
            0,
            stream.native_handle()>>>(input, output, output_elements);
    } else if (weights.format() == core::WeightFormat::Q6K) {
        dequantize_q6k_kernel<<<
            static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(kBlockSize),
            0,
            stream.native_handle()>>>(input, output, output_elements);
    } else {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedDequantizeBackend::dequantize_row format");
    }
    return check_last_cuda_error("quantized row dequantize kernel launch");
}

const CudaQuantizedDequantizeBackend& cuda_quantized_dequantize_backend() {
    static const CudaQuantizedDequantizeBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
