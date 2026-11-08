// Chapter 05：实现 GGML Q4_K super-block 的 CPU reference 解码。
// 约束：严格遵循 256 个值、144 字节的 block 布局。
// 验证入口：chapter05_quantization_test 与真实 TinyLlama 对照。
#include "chapter05_quantization/q4k_dequantize.h"

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

void get_scale_min(const std::uint8_t* scales, int index, std::uint8_t* scale, std::uint8_t* minimum) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

bool fail(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
    return false;
}

}  // namespace

bool dequantize_q4_k(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error) {
    if (data == nullptr || output == nullptr) return fail(error, "Q4_K input and output must not be null");
    if (element_count == 0 || element_count % kQ4KBlockElements != 0) {
        return fail(error, "Q4_K element count must be a non-zero multiple of 256");
    }
    const std::size_t block_count = element_count / kQ4KBlockElements;
    if (block_count > std::numeric_limits<std::size_t>::max() / kQ4KBlockBytes ||
        byte_count != block_count * kQ4KBlockBytes) {
        return fail(error, "Q4_K byte count does not match block count");
    }

    output->assign(element_count, 0.0f);
    for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
        Q4KBlock block{};
        std::memcpy(&block, data + block_index * kQ4KBlockBytes, sizeof(block));
        const float d = half_to_float(block.d);
        const float minimum = half_to_float(block.dmin);
        const std::uint8_t* q = block.qs;
        int scale_index = 0;
        std::size_t output_index = block_index * kQ4KBlockElements;

        for (std::size_t group = 0; group < kQ4KBlockElements; group += 64) {
            std::uint8_t scale = 0;
            std::uint8_t min_value = 0;
            get_scale_min(block.scales, scale_index, &scale, &min_value);
            const float d1 = d * scale;
            const float m1 = minimum * min_value;
            get_scale_min(block.scales, scale_index + 1, &scale, &min_value);
            const float d2 = d * scale;
            const float m2 = minimum * min_value;
            for (int i = 0; i < 32; ++i) {
                (*output)[output_index++] = d1 * (q[i] & 0x0f) - m1;
            }
            for (int i = 0; i < 32; ++i) {
                (*output)[output_index++] = d2 * (q[i] >> 4) - m2;
            }
            q += 32;
            scale_index += 2;
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
