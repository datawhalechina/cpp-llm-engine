// Chapter 05：按 GGML Q2_K 布局恢复 2-bit quant 与 4-bit scale/min。
#include "chapter05_quantization/q2k_dequantize.h"

#include <cstring>

namespace chapter05_quantization {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

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
            std::uint32_t exp = 113;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            result = sign | (exp << 23) |
                ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0F;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

}  // namespace

void decode_q2_k_block(const std::uint8_t* block, float* output) {
    const auto* scales = block;
    const auto* quants = block + 16;
    std::uint16_t scale_bits = 0;
    std::uint16_t minimum_bits = 0;
    std::memcpy(&scale_bits, block + 80, sizeof(scale_bits));
    std::memcpy(&minimum_bits, block + 82, sizeof(minimum_bits));
    const float global_scale = half_to_float(scale_bits);
    const float global_minimum = half_to_float(minimum_bits);

    std::size_t scale_index = 0;
    std::size_t output_index = 0;
    for (std::size_t half = 0; half < 2; ++half) {
        const auto* quant_half = quants + half * 32;
        unsigned shift = 0;
        for (std::size_t group = 0; group < 4; ++group) {
            for (std::size_t lane_group = 0; lane_group < 2; ++lane_group) {
                const std::uint8_t packed_scale = scales[scale_index++];
                const float scale = global_scale *
                    static_cast<float>(packed_scale & 0x0fu);
                const float minimum = global_minimum *
                    static_cast<float>(packed_scale >> 4);
                const std::size_t lane_offset = lane_group * 16;
                for (std::size_t lane = 0; lane < 16; ++lane) {
                    const int quant =
                        (quant_half[lane_offset + lane] >> shift) & 0x03;
                    output[output_index++] =
                        scale * static_cast<float>(quant) - minimum;
                }
            }
            shift += 2;
        }
    }
}

bool dequantize_q2_k(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error) {
    if (data == nullptr || output == nullptr) {
        return fail(error, "Q2_K input and output must not be null");
    }
    if (element_count == 0 || element_count % kQ2KBlockElements != 0) {
        return fail(error,
                    "Q2_K element count must be a non-zero multiple of 256");
    }
    const std::size_t block_count = element_count / kQ2KBlockElements;
    if (byte_count != block_count * kQ2KBlockBytes) {
        return fail(error, "Q2_K byte count does not match block count");
    }
    std::vector<float> decoded(element_count, 0.0F);
    for (std::size_t block = 0; block < block_count; ++block) {
        decode_q2_k_block(data + block * kQ2KBlockBytes,
                          decoded.data() + block * kQ2KBlockElements);
    }
    *output = std::move(decoded);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
