// CUDA Backend：实现 Q4_K/Q6_K 解码与 GEMV 融合 kernel。
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
constexpr std::size_t kBatchTile = 4;

__device__ float half_bits_to_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
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
    std::size_t cols,
    std::size_t batch_size) {
    const std::size_t row = blockIdx.x;
    const std::size_t batch_begin = blockIdx.y * kBatchTile;
    const std::size_t blocks_per_row = cols / core::kQuantizedBlockElements;
    constexpr std::size_t block_bytes =
        Q4K ? core::kQ4KBlockBytes : core::kQ6KBlockBytes;
    const std::uint8_t* row_weights =
        weights + row * blocks_per_row * block_bytes;

    float sums[kBatchTile] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const std::uint8_t* packed = row_weights + block * block_bytes;
        const std::size_t local = threadIdx.x;
        const float weight = Q4K
            ? decode_q4k(packed, local)
            : decode_q6k(packed, local);
        const std::size_t input_col =
            block * core::kQuantizedBlockElements + local;
        for (std::size_t tile_row = 0; tile_row < kBatchTile; ++tile_row) {
            const std::size_t batch = batch_begin + tile_row;
            if (batch < batch_size) {
                sums[tile_row] += weight * input[batch * cols + input_col];
            }
        }
    }

    const unsigned int lane = threadIdx.x % kWarpSize;
    const unsigned int warp = threadIdx.x / kWarpSize;
    __shared__ float warp_sums[kBatchTile][kWarpsPerBlock];
    for (std::size_t tile_row = 0; tile_row < kBatchTile; ++tile_row) {
        for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            sums[tile_row] += __shfl_down_sync(
                0xFFFFFFFFU, sums[tile_row], offset);
        }
        if (lane == 0) warp_sums[tile_row][warp] = sums[tile_row];
    }
    __syncthreads();
    if (warp == 0) {
        for (std::size_t tile_row = 0; tile_row < kBatchTile; ++tile_row) {
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
    const CudaStream& stream) const {
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
    if (weights.cols() % core::kQuantizedBlockElements != 0 ||
        batch_size == 0 || batch_size > 65535 ||
        input_width != weights.cols() || output_width != weights.rows() ||
        weights.rows() > std::numeric_limits<unsigned int>::max()) {
        return CudaStatus::from(
            cudaErrorInvalidValue,
            "CudaQuantizedGemvBackend::compute_batch shape");
    }

    const auto* packed = static_cast<const std::uint8_t*>(weights.data());
    const dim3 grid(
        static_cast<unsigned int>(weights.rows()),
        static_cast<unsigned int>(
            batch_size / kBatchTile + (batch_size % kBatchTile != 0 ? 1 : 0)));
    if (weights.format() == core::WeightFormat::Q4K) {
        quantized_gemv_kernel<true><<<grid, kThreads, 0, stream.native_handle()>>>(
            packed, input, output, weights.cols(), batch_size);
    } else if (weights.format() == core::WeightFormat::Q6K) {
        quantized_gemv_kernel<false><<<grid, kThreads, 0, stream.native_handle()>>>(
            packed, input, output, weights.cols(), batch_size);
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
