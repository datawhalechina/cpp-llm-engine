// Chapter 05：声明 GGML IQ4_NL 非线性 4-bit block 的反量化接口。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chapter05_quantization {

constexpr std::size_t kIQ4NLBlockElements = 32;
constexpr std::size_t kIQ4NLBlockBytes = 18;

void decode_iq4_nl_block(const std::uint8_t* data, float* output);

bool dequantize_iq4_nl(const std::uint8_t* data,
                       std::size_t byte_count,
                       std::size_t element_count,
                       std::vector<float>* output,
                       std::string* error = nullptr);

}  // namespace chapter05_quantization
