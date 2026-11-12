// CUDA Backend：实现 Q2_K/Q3_K/Q4_K/Q6_K/IQ4_NL 解码与 GEMV 融合 kernel。
// 契约：一个 CUDA block 负责同一输出行的一个批次 tile，线程直接累加 packed 权重点积。
// 验证：cuda_quantized_linear_test.cu 与 Chapter05 CPU reference 对照。
#include "cpp_llm_engine/cuda/quantized_gemv_backend.h"

#include "cpp_llm_engine/core/quantization_contract.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_stream.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cpp_llm_engine::cuda {
namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kWarpsPerBlock = kThreads / kWarpSize;

__device__ __constant__ std::int8_t kIQ4NLGemvQuantValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

__device__ float half_bits_to_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
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

__device__ float decode_q2k(
    const std::uint8_t* block,
    std::size_t local) {
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
    return global_scale * static_cast<float>(packed_scale & 0x0f) *
            static_cast<float>(quant) -
        global_minimum * static_cast<float>(packed_scale >> 4);
}

__device__ float decode_q3k(
    const std::uint8_t* block,
    std::size_t local) {
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
    return half_bits_to_float(load_u16(block + 108)) *
        static_cast<float>(scale * quant);
}

__device__ float decode_iq4nl(
    const std::uint8_t* block,
    std::size_t local) {
    const float scale = half_bits_to_float(load_u16(block));
    const std::size_t quant_index = local % 16;
    const std::uint8_t packed = block[2 + quant_index];
    const std::uint8_t quant = local < 16 ? packed & 0x0f : packed >> 4;
    return scale * static_cast<float>(kIQ4NLGemvQuantValues[quant]);
}

template <core::WeightFormat Format, std::size_t BatchTile>
__global__ void quantized_gemv_kernel(
    const std::uint8_t* weights,
    const float* input,
    float* output,
    std::size_t cols,
    std::size_t batch_size) {
    const std::size_t row = blockIdx.x;
    const std::size_t batch_begin = blockIdx.y * BatchTile;
    constexpr std::size_t block_elements =
        Format == core::WeightFormat::IQ4_NL
            ? core::kIQ4NLBlockElements
            : core::kQuantizedBlockElements;
    const std::size_t blocks_per_row = cols / block_elements;
    constexpr std::size_t block_bytes =
        Format == core::WeightFormat::Q4K ? core::kQ4KBlockBytes :
        Format == core::WeightFormat::Q6K ? core::kQ6KBlockBytes :
        Format == core::WeightFormat::Q2K ? core::kQ2KBlockBytes :
        Format == core::WeightFormat::Q3K ? core::kQ3KBlockBytes :
        core::kIQ4NLBlockBytes;
    const std::uint8_t* row_weights =
        weights + row * blocks_per_row * block_bytes;

    float sums[BatchTile] = {};
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* packed = row_weights + block * block_bytes;
        const std::size_t local = threadIdx.x;
        if (local < block_elements) {
            float weight = 0.0F;
            if constexpr (Format == core::WeightFormat::Q4K) {
                weight = decode_q4k(packed, local);
            } else if constexpr (Format == core::WeightFormat::Q6K) {
                weight = decode_q6k(packed, local);
            } else if constexpr (Format == core::WeightFormat::Q2K) {
                weight = decode_q2k(packed, local);
            } else if constexpr (Format == core::WeightFormat::Q3K) {
                weight = decode_q3k(packed, local);
            } else {
                weight = decode_iq4nl(packed, local);
            }
            const std::size_t input_col = block * block_elements + local;
            for (std::size_t tile_row = 0; tile_row < BatchTile; ++tile_row) {
                const std::size_t batch = batch_begin + tile_row;
                if (batch < batch_size) {
                    sums[tile_row] += weight * input[batch * cols + input_col];
                }
            }
        }
    }

    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    __shared__ float warp_sums[BatchTile][kWarpsPerBlock];
    for (std::size_t tile_row = 0; tile_row < BatchTile; ++tile_row) {
        for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            sums[tile_row] += __shfl_down_sync(
                0xFFFFFFFFU, sums[tile_row], offset);
        }
        if (lane == 0) warp_sums[tile_row][warp] = sums[tile_row];
    }
    __syncthreads();
    if (warp == 0) {
        for (std::size_t tile_row = 0; tile_row < BatchTile; ++tile_row) {
            float total = lane < kWarpsPerBlock ? warp_sums[tile_row][lane] : 0.0F;
            for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
                total += __shfl_down_sync(0xFFFFFFFFU, total, offset);
            }
            const std::size_t batch = batch_begin + tile_row;
            if (lane == 0 && batch < batch_size) {
                output[batch * gridDim.x + row] = total;
            }
        }
    }
}

template <core::WeightFormat Format, std::size_t BatchTile>
void launch_quantized_gemv(
    const std::uint8_t* packed,
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    std::size_t batch_size,
    cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(
            batch_size / BatchTile + (batch_size % BatchTile != 0 ? 1 : 0)));
    quantized_gemv_kernel<Format, BatchTile><<<grid, kThreads, 0, stream>>>(
        packed, input, output, cols, batch_size);
}

template <core::WeightFormat Format>
void launch_quantized_gemv_path(
    CudaQuantizedGemvPath path,
    const std::uint8_t* packed,
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    std::size_t batch_size,
    cudaStream_t stream) {
    if (path == CudaQuantizedGemvPath::Adaptive && batch_size == 1) {
        launch_quantized_gemv<Format, 1>(
            packed, input, output, rows, cols, batch_size, stream);
    } else if (path == CudaQuantizedGemvPath::Adaptive && batch_size == 2) {
        launch_quantized_gemv<Format, 2>(
            packed, input, output, rows, cols, batch_size, stream);
    } else {
        launch_quantized_gemv<Format, 4>(
            packed, input, output, rows, cols, batch_size, stream);
    }
}

}  // namespace

core::BackendKind CudaQuantizedGemvBackend::kind() const {
    return core::BackendKind::CUDA;
}

std::string_view CudaQuantizedGemvBackend::name() const {
    return "cuda-packed-fused-gemv";
}

CudaStatus CudaQuantizedGemvBackend::compute(
    const CudaQuantizedMatrix& weights,
    const float* input,
    std::size_t input_elements,
    float* output,
    std::size_t output_elements,
    const CudaStream& stream) const {
    return compute_batch(
        weights, input, 1, input_elements, output, output_elements, stream);
}

CudaStatus CudaQuantizedGemvBackend::compute_batch(
    const CudaQuantizedMatrix& weights,
    const float* input,
    std::size_t batch_size,
    std::size_t input_width,
    float* output,
    std::size_t output_width,
    const CudaStream& stream,
    CudaQuantizedGemvPath path) const {
    if (weights.empty() || input == nullptr || output == nullptr) {
        return CudaStatus::from(
            cudaErrorInvalidDevicePointer,
            "CudaQuantizedGemvBackend::compute_batch pointers");
    }
    if (!stream.valid()) {
        return CudaStatus::from(
            cudaErrorInvalidResourceHandle,
            "CudaQuantizedGemvBackend::compute_batch stream");
    }
    std::size_t block_elements = 0;
    std::size_t block_bytes = 0;
    const core::Status layout_status = core::quantized_block_layout(
        weights.format(), &block_elements, &block_bytes);
    (void)block_bytes;
    if (!layout_status.is_ok() || weights.cols() % block_elements != 0 ||
        batch_size == 0 || batch_size > 65535 ||
        input_width != weights.cols() || output_width != weights.rows() ||
        weights.rows() > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedGemvBackend::compute_batch shape");
    }

    const auto* packed = static_cast<const std::uint8_t*>(weights.data());
    if (weights.format() == core::WeightFormat::Q4K) {
        launch_quantized_gemv_path<core::WeightFormat::Q4K>(
            path, packed, input, output, weights.rows(), weights.cols(),
            batch_size, stream.native_handle());
    } else if (weights.format() == core::WeightFormat::Q6K) {
        launch_quantized_gemv_path<core::WeightFormat::Q6K>(
            path, packed, input, output, weights.rows(), weights.cols(),
            batch_size, stream.native_handle());
    } else if (weights.format() == core::WeightFormat::Q2K) {
        launch_quantized_gemv_path<core::WeightFormat::Q2K>(
            path, packed, input, output, weights.rows(), weights.cols(),
            batch_size, stream.native_handle());
    } else if (weights.format() == core::WeightFormat::Q3K) {
        launch_quantized_gemv_path<core::WeightFormat::Q3K>(
            path, packed, input, output, weights.rows(), weights.cols(),
            batch_size, stream.native_handle());
    } else if (weights.format() == core::WeightFormat::IQ4_NL) {
        launch_quantized_gemv_path<core::WeightFormat::IQ4_NL>(
            path, packed, input, output, weights.rows(), weights.cols(),
            batch_size, stream.native_handle());
    } else {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedGemvBackend::compute format");
    }
    return check_last_cuda_error("quantized fused batch GEMV kernel launch");
}

const CudaQuantizedGemvBackend& cuda_quantized_gemv_backend() {
    static const CudaQuantizedGemvBackend backend;
    return backend;
}

}  // namespace cpp_llm_engine::cuda
