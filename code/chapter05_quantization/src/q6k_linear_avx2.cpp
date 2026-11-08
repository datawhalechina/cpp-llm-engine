// Chapter 05：Q6_K fused AVX2/FMA Linear 实现。
// 职责：按 8 个元素解码 Q6_K 位布局，并直接累加 W * input。
// 验证：与 scalar fused 输出对照，并由真实 Decoder 的 llama.cpp 数值指标检查。
#include "chapter05_quantization/q6k_linear.h"

#include "chapter05_quantization/q6k_dequantize.h"

#include <immintrin.h>

#include <cstring>
#include <limits>

namespace chapter05_quantization {
namespace {

struct Q6KBlock {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    std::uint16_t d;
};

static_assert(sizeof(Q6KBlock) == kQ6KBlockBytes, "unexpected Q6_K block size");

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
    if (request.weight_format != cpp_llm_engine::core::WeightFormat::Q6K) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K AVX2 backend received an unsupported weight format");
    }
    if (request.weights == nullptr || request.input == nullptr || request.output == nullptr) {
        return cpp_llm_engine::core::Status::error("Q6_K AVX2 buffers must not be null");
    }
    if (request.rows == 0 || request.cols == 0 ||
        request.cols % kQ6KBlockElements != 0) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K AVX2 dimensions must be non-zero and cols must be a multiple of 256");
    }
    if (request.input_size != request.cols || request.output_size < request.rows) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K AVX2 buffer sizes do not match shape");
    }
    const std::size_t blocks_per_row = request.cols / kQ6KBlockElements;
    if (request.rows > std::numeric_limits<std::size_t>::max() / blocks_per_row) {
        return cpp_llm_engine::core::Status::error("Q6_K AVX2 payload is too large");
    }
    const std::size_t block_count = request.rows * blocks_per_row;
    if (block_count > std::numeric_limits<std::size_t>::max() / kQ6KBlockBytes ||
        request.weight_bytes != block_count * kQ6KBlockBytes) {
        return cpp_llm_engine::core::Status::error(
            "Q6_K AVX2 byte count does not match shape");
    }
    return cpp_llm_engine::core::Status::ok();
}

}  // namespace

cpp_llm_engine::core::BackendKind Q6KLinearAvx2Backend::kind() const {
    return cpp_llm_engine::core::BackendKind::CPU;
}

std::string_view Q6KLinearAvx2Backend::name() const {
    return "q6k-fused-avx2";
}

cpp_llm_engine::core::Status Q6KLinearAvx2Backend::compute(
    const cpp_llm_engine::core::LinearRequest& request) const {
    const auto status = validate(request);
    if (!status.is_ok()) return status;

    const auto* weights = static_cast<const std::uint8_t*>(request.weights);
    const std::size_t blocks_per_row = request.cols / kQ6KBlockElements;
    const __m256i low_mask = _mm256_set1_epi32(0x0f);
    const __m256i high_mask = _mm256_set1_epi32(0x03);
    const __m256i zero_point = _mm256_set1_epi32(32);

    for (std::size_t row = 0; row < request.rows; ++row) {
        __m256 accumulator = _mm256_setzero_ps();
        const auto* row_weights = weights + row * blocks_per_row * kQ6KBlockBytes;
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            Q6KBlock block{};
            std::memcpy(&block, row_weights + block_index * kQ6KBlockBytes,
                        sizeof(block));
            const float d = half_to_float(block.d);
            const std::size_t input_offset = block_index * kQ6KBlockElements;
            for (std::size_t half = 0; half < kQ6KBlockElements; half += 128) {
                const auto* ql = block.ql + half / 2;
                const auto* qh = block.qh + half / 4;
                const auto* scales = block.scales + half / 16;
                for (std::size_t l = 0; l < 32; l += 8) {
                    const __m256i ql_low = load_u8x8(ql + l);
                    const __m256i ql_high = load_u8x8(ql + l + 32);
                    const __m256i qh_values = load_u8x8(qh + l);
                    const __m256i q1 = _mm256_sub_epi32(
                        _mm256_or_si256(
                            _mm256_and_si256(ql_low, low_mask),
                            _mm256_slli_epi32(
                                _mm256_and_si256(qh_values, high_mask), 4)),
                        zero_point);
                    const __m256i q2 = _mm256_sub_epi32(
                        _mm256_or_si256(
                            _mm256_and_si256(ql_high, low_mask),
                            _mm256_slli_epi32(
                                _mm256_and_si256(
                                    _mm256_srli_epi32(qh_values, 2), high_mask), 4)),
                        zero_point);
                    const __m256i q3 = _mm256_sub_epi32(
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi32(ql_low, 4), low_mask),
                            _mm256_slli_epi32(
                                _mm256_and_si256(
                                    _mm256_srli_epi32(qh_values, 4), high_mask), 4)),
                        zero_point);
                    const __m256i q4 = _mm256_sub_epi32(
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi32(ql_high, 4), low_mask),
                            _mm256_slli_epi32(
                                _mm256_and_si256(
                                    _mm256_srli_epi32(qh_values, 6), high_mask), 4)),
                        zero_point);
                    const std::size_t scale_index = l / 16;
                    const __m256 w1 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q1),
                        _mm256_set1_ps(d * scales[scale_index + 0]));
                    const __m256 w2 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q2),
                        _mm256_set1_ps(d * scales[scale_index + 2]));
                    const __m256 w3 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q3),
                        _mm256_set1_ps(d * scales[scale_index + 4]));
                    const __m256 w4 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q4),
                        _mm256_set1_ps(d * scales[scale_index + 6]));
                    accumulator = _mm256_fmadd_ps(
                        w1, _mm256_loadu_ps(request.input + input_offset + half + l),
                        accumulator);
                    accumulator = _mm256_fmadd_ps(
                        w2, _mm256_loadu_ps(request.input + input_offset + half + l + 32),
                        accumulator);
                    accumulator = _mm256_fmadd_ps(
                        w3, _mm256_loadu_ps(request.input + input_offset + half + l + 64),
                        accumulator);
                    accumulator = _mm256_fmadd_ps(
                        w4, _mm256_loadu_ps(request.input + input_offset + half + l + 96),
                        accumulator);
                }
            }
        }
        request.output[row] = horizontal_sum(accumulator);
    }
    return cpp_llm_engine::core::Status::ok();
}

const Q6KLinearAvx2Backend& q6k_avx2_backend() {
    static const Q6KLinearAvx2Backend backend;
    return backend;
}

bool Q6KLinearAvx2::forward(const std::uint8_t* weights,
                            std::size_t weight_bytes,
                            std::size_t rows,
                            std::size_t cols,
                            const std::vector<float>& input,
                            std::vector<float>* output,
                            std::string* error) {
    if (output == nullptr) {
        if (error != nullptr) *error = "Q6_K AVX2 output must not be null";
        return false;
    }
    output->assign(rows, 0.0f);
    const cpp_llm_engine::core::LinearRequest request{
        weights, weight_bytes, input.data(), input.size(), output->data(), output->size(),
        rows, cols, cpp_llm_engine::core::WeightFormat::Q6K};
    const auto status = q6k_avx2_backend().compute(request);
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
