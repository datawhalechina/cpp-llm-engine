// Chapter 05：Q4_K fused AVX2/FMA Linear 实现。
// 职责：直接从 block 解码 4-bit 权重，并按 8 个值累加 W * input。
// 验证：与 scalar reference 对照，并由真实 Decoder golden 和 A/B 检查。
#include "chapter05_quantization/q4k_linear.h"

#include "chapter05_quantization/q4k_dequantize.h"

#include <immintrin.h>

#include <cstring>
#include <limits>

namespace chapter05_quantization {
namespace {

struct Q4KBlock {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q4KBlock) == kQ4KBlockBytes, "unexpected Q4_K block size");

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::uint32_t normalized = mantissa;
            std::uint32_t exp = 127 - 14;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            result = sign | (exp << 23) | ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

void get_scale_min(const std::uint8_t* scales,
                   int index,
                   std::uint8_t* scale,
                   std::uint8_t* minimum) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    const __m128 pair = _mm_add_ps(low, high);
    const __m128 shuffled = _mm_movehdup_ps(pair);
    const __m128 summed = _mm_add_ps(pair, shuffled);
    const __m128 high_sum = _mm_movehl_ps(shuffled, summed);
    return _mm_cvtss_f32(_mm_add_ss(summed, high_sum));
}

__m256i load_u8x8(const std::uint8_t* source) {
    return _mm256_cvtepu8_epi32(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source)));
}

cpp_llm_engine::core::Status validate(
    const cpp_llm_engine::core::LinearRequest& request) {
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q4K) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K AVX2 backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q4_K AVX2 buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 ||
        request.cols % kQ4KBlockElements != 0) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K AVX2 dimensions must be non-zero and cols must be a multiple of 256");
    }
    if (request.input_size != request.cols || request.output_size < request.rows) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K AVX2 buffer sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ4KBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row) {
        return cpp_llm_engine::core::Status::error("Q4_K AVX2 payload is too large");
    }
    const std::size_t block_count = request.rows * blocks_per_row;
    if (block_count > std::numeric_limits<std::size_t>::max() / kQ4KBlockBytes ||
        request.weight_bytes != block_count * kQ4KBlockBytes) {
        return cpp_llm_engine::core::Status::error(
            "Q4_K AVX2 byte count does not match shape");
    }
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace

cpp_llm_engine::core::BackendKind Q4KLinearAvx2Backend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q4KLinearAvx2Backend::name() const {
    return "q4k-fused-avx2";
}

cpp_llm_engine::core::Status Q4KLinearAvx2Backend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    const auto status = validate(request);
    if (!status.is_ok()) return status;

    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    const std::size_t blocks_per_row = request.cols / kQ4KBlockElements;
    const __m256i low_mask = _mm256_set1_epi32(0x0f);
    for (std::size_t row = 0; row < request.rows; ++row) {
        __m256 accumulator = _mm256_setzero_ps();
        const auto* row_weights = weights + row * blocks_per_row * kQ4KBlockBytes;
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            Q4KBlock block{};
            std::memcpy(&block, row_weights + block_index * kQ4KBlockBytes,
                        sizeof(block));
            const float d = half_to_float(block.d);
            const float minimum = half_to_float(block.dmin);
            const std::size_t input_offset = block_index * kQ4KBlockElements;
            const auto* quants = block.qs;
            int scale_index = 0;
            for (std::size_t group = 0; group < kQ4KBlockElements; group += 64) {
                std::uint8_t scale = 0;
                std::uint8_t min_value = 0;
                get_scale_min(block.scales, scale_index, &scale, &min_value);
                const float d1 = d * scale;
                const float m1 = minimum * min_value;
                get_scale_min(block.scales, scale_index + 1, &scale, &min_value);
                const float d2 = d * scale;
                const float m2 = minimum * min_value;
                for (std::size_t i = 0; i < 32; i += 8) {
                    const __m256i packed = load_u8x8(quants + i);
                    const __m256 low = _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_cvtepi32_ps(_mm256_and_si256(packed, low_mask)),
                            _mm256_set1_ps(d1)),
                        _mm256_set1_ps(m1));
                    const __m256 high = _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_cvtepi32_ps(_mm256_srli_epi32(packed, 4)),
                            _mm256_set1_ps(d2)),
                        _mm256_set1_ps(m2));
                    accumulator = _mm256_fmadd_ps(
                        low,
                        _mm256_loadu_ps(request.input + input_offset + group + i),
                        accumulator);
                    accumulator = _mm256_fmadd_ps(
                        high,
                        _mm256_loadu_ps(
                            request.input + input_offset + group + 32 + i),
                        accumulator);
                }
                quants += 32;
                scale_index += 2;
            }
        }
        request.output[row] = horizontal_sum(accumulator);
    }
    return cpp_llm_engine::core::Status::ok();
}

const Q4KLinearAvx2Backend& q4k_avx2_backend() {
    static const Q4KLinearAvx2Backend backend;
    return backend;
}

bool Q4KLinearAvx2::forward(const std::uint8_t* weights,
                            std::size_t weight_bytes,
                            std::size_t rows,
                            std::size_t cols,
                            const std::vector<float>& input,
                            std::vector<float>* output,
                            std::string* error) {
    if (output == nullptr) {
        if (error != nullptr) *error = "Q4_K AVX2 output must not be null";
        return false;
    }
    output->assign(rows, 0.0f);
    const cpp_llm_engine::core::LinearRequest request{
        weights, weight_bytes, input.data(), input.size(), output->data(), output->size(),
        rows, cols, cpp_llm_engine::core::WeightFormat::Q4K};
    const auto status = q4k_avx2_backend().compute(request);
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
