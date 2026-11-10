// Chapter 05：按 GGML Q3_K 布局恢复 3-bit quant 和 6-bit group scale。
#include "chapter05_quantization/q3k_dequantize.h"

#include <array>
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

}  // namespace

void decode_q3_k_block(const std::uint8_t* block, float* output) {
    const auto* hmask = block;
    const auto* quants = block + 32;
    const auto* packed_scales = block + 96;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, block + 108, sizeof(scale_bits));

    std::array<std::uint32_t, 4> aux{};
    std::memcpy(aux.data(), packed_scales, 12);
    constexpr std::uint32_t kMask2 = 0x03030303u;
    constexpr std::uint32_t kMask4 = 0x0f0f0f0fu;
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kMask4) | (((tmp >> 4) & kMask2) << 4);
    aux[3] = ((aux[1] >> 4) & kMask4) | (((tmp >> 6) & kMask2) << 4);
    aux[0] = (aux[0] & kMask4) | (((tmp >> 0) & kMask2) << 4);
    aux[1] = (aux[1] & kMask4) | (((tmp >> 2) & kMask2) << 4);

    const auto* scales = reinterpret_cast<const std::uint8_t*>(aux.data());
    const float global_scale = half_to_float(scale_bits);
    std::size_t scale_index = 0;
    std::size_t output_index = 0;
    std::uint8_t high_mask = 1;
    for (std::size_t half = 0; half < 2; ++half) {
        const auto* quant_half = quants + half * 32;
        for (std::size_t group = 0; group < 4; ++group) {
            const unsigned shift = static_cast<unsigned>(group * 2);
            for (std::size_t lane_group = 0; lane_group < 2; ++lane_group) {
                const float scale = global_scale *
                    static_cast<float>(static_cast<int>(scales[scale_index++]) - 32);
                const std::size_t lane_offset = lane_group * 16;
                for (std::size_t lane = 0; lane < 16; ++lane) {
                    const std::size_t index = lane_offset + lane;
                    const int low = (quant_half[index] >> shift) & 0x03;
                    const int quant = low - ((hmask[index] & high_mask) ? 0 : 4);
                    output[output_index++] = scale * static_cast<float>(quant);
                }
            }
            high_mask = static_cast<std::uint8_t>(high_mask << 1);
        }
    }
}

bool dequantize_q3_k(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error) {
    if (data == nullptr || output == nullptr) {
        return fail(error, "Q3_K input and output must not be null");
    }
    if (element_count == 0 || element_count % kQ3KBlockElements != 0) {
        return fail(error, "Q3_K element count must be a non-zero multiple of 256");
    }
    const std::size_t block_count = element_count / kQ3KBlockElements;
    if (byte_count != block_count * kQ3KBlockBytes) {
        return fail(error, "Q3_K byte count does not match block count");
    }
    std::vector<float> decoded(element_count);
    for (std::size_t block = 0; block < block_count; ++block) {
        decode_q3_k_block(data + block * kQ3KBlockBytes,
                          decoded.data() + block * kQ3KBlockElements);
    }
    *output = std::move(decoded);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
