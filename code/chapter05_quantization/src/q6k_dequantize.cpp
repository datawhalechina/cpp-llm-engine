// 中文：按 llama.cpp Q6_K 布局解码 GGUF payload，提供后续 Linear 接入的正确性基线。
#include "chapter05_quantization/q6k_dequantize.h"

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

bool dequantize_q6_k(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error) {
    if (data == nullptr || output == nullptr) return fail(error, "Q6_K input and output must not be null");
    if (element_count == 0 || element_count % kQ6KBlockElements != 0) {
        return fail(error, "Q6_K element count must be a non-zero multiple of 256");
    }
    const std::size_t block_count = element_count / kQ6KBlockElements;
    if (block_count > std::numeric_limits<std::size_t>::max() / kQ6KBlockBytes ||
        byte_count != block_count * kQ6KBlockBytes) {
        return fail(error, "Q6_K byte count does not match block count");
    }

    output->assign(element_count, 0.0f);
    for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
        Q6KBlock block{};
        std::memcpy(&block, data + block_index * kQ6KBlockBytes, sizeof(block));
        const float d = half_to_float(block.d);
        const std::uint8_t* ql = block.ql;
        const std::uint8_t* qh = block.qh;
        const std::int8_t* scales = block.scales;
        std::size_t output_index = block_index * kQ6KBlockElements;

        for (std::size_t half = 0; half < kQ6KBlockElements; half += 128) {
            for (int l = 0; l < 32; ++l) {
                const int scale_index = l / 16;
                const int q1 = static_cast<int>((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) - 32;
                const int q2 = static_cast<int>((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
                const int q3 = static_cast<int>(((ql[l] >> 4) & 0x0f) | (((qh[l] >> 4) & 0x03) << 4)) - 32;
                const int q4 = static_cast<int>(((ql[l + 32] >> 4) & 0x0f) | (((qh[l] >> 6) & 0x03) << 4)) - 32;
                (*output)[output_index + l] = d * scales[scale_index + 0] * q1;
                (*output)[output_index + l + 32] = d * scales[scale_index + 2] * q2;
                (*output)[output_index + l + 64] = d * scales[scale_index + 4] * q3;
                (*output)[output_index + l + 96] = d * scales[scale_index + 6] * q4;
            }
            output_index += 128;
            ql += 64;
            qh += 32;
            scales += 8;
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace chapter05_quantization
