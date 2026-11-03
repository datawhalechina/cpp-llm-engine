// Chapter 05：实现 GGML Q8_0 block 的 CPU reference 解码。
// 约束：严格遵循 32 个值、34 字节的 block 布局，并按 d * q 还原 Float32。
// 验证：chapter05_quantization_test 与多模态真实首 Block 检查。
#include "chapter05_quantization/q8_0_dequantize.h"

#include <cstring>
#include <limits>

namespace chapter05_quantization {
namespace {

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
            normalized &= 0x03ffu;
            result = sign | (exp << 23) | (normalized << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

bool fail(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
    return false;
}

}  // namespace

bool dequantize_q8_0(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error) {
    if (data == nullptr || output == nullptr) {
        return fail(error, "Q8_0 input and output must not be null");
    }
    if (element_count == 0 || element_count % kQ8_0BlockElements != 0) {
        return fail(error, "Q8_0 element count must be a non-zero multiple of 32");
    }
    const std::size_t block_count = element_count / kQ8_0BlockElements;
    if (block_count > std::numeric_limits<std::size_t>::max() / kQ8_0BlockBytes ||
        byte_count != block_count * kQ8_0BlockBytes) {
        return fail(error, "Q8_0 byte count does not match block count");
    }

    output->resize(element_count);
    for (std::size_t block = 0; block < block_count; ++block) {
        const std::uint8_t* source = data + block * kQ8_0BlockBytes;
        std::uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, source, sizeof(scale_bits));
        const float scale = half_to_float(scale_bits);
        for (std::size_t value = 0; value < kQ8_0BlockElements; ++value) {
            std::int8_t quant = 0;
            std::memcpy(&quant, source + sizeof(scale_bits) + value, sizeof(quant));
            (*output)[block * kQ8_0BlockElements + value] =
                scale * static_cast<float>(quant);
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
