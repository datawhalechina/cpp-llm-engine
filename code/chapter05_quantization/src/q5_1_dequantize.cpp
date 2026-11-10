// Chapter 05：按 GGML Q5_1 布局恢复 5-bit quant、scale 和 minimum。
#include "chapter05_quantization/q5_1_dequantize.h"

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

void decode_q5_1_block(const std::uint8_t* block, float* output) {
    std::uint16_t scale_bits = 0;
    std::uint16_t minimum_bits = 0;
    std::uint32_t high_bits = 0;
    std::memcpy(&scale_bits, block, sizeof(scale_bits));
    std::memcpy(&minimum_bits, block + 2, sizeof(minimum_bits));
    std::memcpy(&high_bits, block + 4, sizeof(high_bits));
    const auto* quants = block + 8;
    const float scale = half_to_float(scale_bits);
    const float minimum = half_to_float(minimum_bits);
    for (std::size_t lane = 0; lane < 16; ++lane) {
        const int low = quants[lane] & 0x0f;
        const int high = quants[lane] >> 4;
        const int q0 = low | static_cast<int>(((high_bits >> lane) << 4) & 0x10u);
        const int q1 = high |
            static_cast<int>(((high_bits >> (lane + 12))     ) & 0x10u);
        output[lane] = scale * static_cast<float>(q0) + minimum;
        output[lane + 16] = scale * static_cast<float>(q1) + minimum;
    }
}

bool dequantize_q5_1(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error) {
    if (data == nullptr || output == nullptr) {
        return fail(error, "Q5_1 input and output must not be null");
    }
    if (element_count == 0 || element_count % kQ5_1BlockElements != 0) {
        return fail(error, "Q5_1 element count must be a non-zero multiple of 32");
    }
    const std::size_t block_count = element_count / kQ5_1BlockElements;
    if (byte_count != block_count * kQ5_1BlockBytes) {
        return fail(error, "Q5_1 byte count does not match block count");
    }
    std::vector<float> decoded(element_count);
    for (std::size_t block = 0; block < block_count; ++block) {
        decode_q5_1_block(data + block * kQ5_1BlockBytes,
                          decoded.data() + block * kQ5_1BlockElements);
    }
    *output = std::move(decoded);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
