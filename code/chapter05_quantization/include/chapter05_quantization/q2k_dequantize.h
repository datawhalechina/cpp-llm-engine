// Chapter 05：声明 GGML Q2_K super-block 的 CPU 参考反量化接口。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chapter05_quantization {

constexpr std::size_t kQ2KBlockElements = 256;
constexpr std::size_t kQ2KBlockBytes = 84;

void decode_q2_k_block(const std::uint8_t* data, float* output);

bool dequantize_q2_k(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error = nullptr);

}  // namespace chapter05_quantization
