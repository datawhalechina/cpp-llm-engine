// CUDA Backend：实现 GGML Q2_K/Q3_K/Q4_K/Q6_K/IQ4_NL block 到 Float32 workspace 的解码。
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

__device__ __constant__ std::int8_t kIQ4NLQuantValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

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

__device__ std::uint32_t load_u32(const std::uint8_t* data) {
    return static_cast<std::uint32_t>(data[0]) |
        (static_cast<std::uint32_t>(data[1]) << 8) |
        (static_cast<std::uint32_t>(data[2]) << 16) |
        (static_cast<std::uint32_t>(data[3]) << 24);
}

__device__ std::uint8_t q3k_scale(const std::uint8_t* packed, std::size_t index) {
    constexpr std::uint32_t kMask2 = 0x03030303u;
    constexpr std::uint32_t kMask4 = 0x0f0f0f0fu;
    const std::uint32_t first = load_u32(packed);
    const std::uint32_t second = load_u32(packed + 4);
    const std::uint32_t third = load_u32(packed + 8);
    std::uint32_t word = 0;
    if (index / 4 == 0) {
        word = (first & kMask4) | (((third >> 0) & kMask2) << 4);
    } else if (index / 4 == 1) {
        word = (second & kMask4) | (((third >> 2) & kMask2) << 4);
    } else if (index / 4 == 2) {
        word = ((first >> 4) & kMask4) | (((third >> 4) & kMask2) << 4);
    } else {
        word = ((second >> 4) & kMask4) | (((third >> 6) & kMask2) << 4);
    }
    return static_cast<std::uint8_t>((word >> ((index % 4) * 8)) & 0xffu);
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

__global__ void dequantize_q2k_kernel(
    const std::uint8_t* input,
    float* output,
    std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const std::size_t block_index = index / core::kQuantizedBlockElements;
    const std::size_t local = index % core::kQuantizedBlockElements;
    const std::uint8_t* block = input + block_index * core::kQ2KBlockBytes;
    const std::size_t half = local / 128;
    const std::size_t within_half = local % 128;
    const std::size_t group = within_half / 32;
    const std::size_t lane_group = (within_half % 32) / 16;
    const std::size_t lane = within_half % 16;
    const std::size_t scale_index = half * 8 + group * 2 + lane_group;
    const std::uint8_t packed_scale = block[scale_index];
    const float global_scale = half_bits_to_float(load_u16(block + 80));
    const float global_minimum = half_bits_to_float(load_u16(block + 82));
    const std::uint8_t packed_quant =
        block[16 + half * 32 + lane_group * 16 + lane];
    const int quant = (packed_quant >> (group * 2)) & 0x03;
    output[index] =
        global_scale * static_cast<float>(packed_scale & 0x0f) *
            static_cast<float>(quant) -
        global_minimum * static_cast<float>(packed_scale >> 4);
}

__global__ void dequantize_q3k_kernel(
    const std::uint8_t* input,
    float* output,
    std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const std::size_t block_index = index / core::kQuantizedBlockElements;
    const std::size_t local = index % core::kQuantizedBlockElements;
    const std::uint8_t* block = input + block_index * core::kQ3KBlockBytes;
    const std::size_t half = local / 128;
    const std::size_t within_half = local % 128;
    const std::size_t group = within_half / 32;
    const std::size_t lane_group = (within_half % 32) / 16;
    const std::size_t lane = within_half % 16;
    const std::size_t scale_index = half * 8 + group * 2 + lane_group;
    const int scale = static_cast<int>(q3k_scale(block + 96, scale_index)) - 32;
    const std::size_t quant_index = lane_group * 16 + lane;
    const std::uint8_t packed = block[32 + half * 32 + quant_index];
    const int low = (packed >> (group * 2)) & 0x03;
    const std::uint8_t high_mask = static_cast<std::uint8_t>(
        1u << static_cast<unsigned int>(half * 4 + group));
    const int quant = low - ((block[quant_index] & high_mask) != 0 ? 0 : 4);
    output[index] = half_bits_to_float(load_u16(block + 108)) *
        static_cast<float>(scale * quant);
}

__global__ void dequantize_iq4nl_kernel(
    const std::uint8_t* input,
    float* output,
    std::size_t elements) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const std::size_t block_index = index / core::kIQ4NLBlockElements;
    const std::size_t local = index % core::kIQ4NLBlockElements;
    const std::uint8_t* block = input + block_index * core::kIQ4NLBlockBytes;
    const float scale = half_bits_to_float(load_u16(block));
    const std::size_t quant_index = local % 16;
    const std::uint8_t packed = block[2 + quant_index];
    const std::uint8_t quant = local < 16 ? packed & 0x0f : packed >> 4;
    output[index] = scale * static_cast<float>(kIQ4NLQuantValues[quant]);
}

}  // namespace

core::BackendKind CudaQuantizedDequantizeBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaQuantizedDequantizeBackend::name() const {
    return "cuda-packed-dequantize";
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
    } else if (weights.format() == core::WeightFormat::Q2K) {
        dequantize_q2k_kernel<<<
            static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(kBlockSize),
            0,
            stream.native_handle()>>>(input, output, output_elements);
    } else if (weights.format() == core::WeightFormat::Q3K) {
        dequantize_q3k_kernel<<<
            static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(kBlockSize),
            0,
            stream.native_handle()>>>(input, output, output_elements);
    } else if (weights.format() == core::WeightFormat::IQ4_NL) {
        dequantize_iq4nl_kernel<<<
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
    } else if (weights.format() == core::WeightFormat::Q2K) {
        dequantize_q2k_kernel<<<
            static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(kBlockSize),
            0,
            stream.native_handle()>>>(input, output, output_elements);
    } else if (weights.format() == core::WeightFormat::Q3K) {
        dequantize_q3k_kernel<<<
            static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(kBlockSize),
            0,
            stream.native_handle()>>>(input, output, output_elements);
    } else if (weights.format() == core::WeightFormat::IQ4_NL) {
        dequantize_iq4nl_kernel<<<
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
