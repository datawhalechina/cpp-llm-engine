// Chapter 05：按 GGML IQ4_NL 查找表恢复非均匀 4-bit quant。
#include "chapter05_quantization/iq4nl_dequantize.h"

#include <array>
#include <cstring>

namespace chapter05_quantization {
namespace {

constexpr std::array<std::int8_t, 16> kQuantValues{
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

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

void decode_iq4_nl_block(const std::uint8_t* block, float* output) {
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, block, sizeof(scale_bits));
    const float scale = half_to_float(scale_bits);
    const auto* quants = block + 2;
    for (std::size_t index = 0; index < 16; ++index) {
        output[index] = scale *
            static_cast<float>(kQuantValues[quants[index] & 0x0fu]);
        output[index + 16] = scale *
            static_cast<float>(kQuantValues[quants[index] >> 4]);
    }
}

bool dequantize_iq4_nl(const std::uint8_t* data,
                       std::size_t byte_count,
                       std::size_t element_count,
                       std::vector<float>* output,
                       std::string* error) {
    if (data == nullptr || output == nullptr) {
        return fail(error, "IQ4_NL input and output must not be null");
    }
    if (element_count == 0 || element_count % kIQ4NLBlockElements != 0) {
        return fail(error,
                    "IQ4_NL element count must be a non-zero multiple of 32");
    }
    const std::size_t block_count = element_count / kIQ4NLBlockElements;
    if (byte_count != block_count * kIQ4NLBlockBytes) {
        return fail(error, "IQ4_NL byte count does not match block count");
    }
    std::vector<float> decoded(element_count, 0.0F);
    for (std::size_t block = 0; block < block_count; ++block) {
        decode_iq4_nl_block(data + block * kIQ4NLBlockBytes,
                            decoded.data() + block * kIQ4NLBlockElements);
    }
    *output = std::move(decoded);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
