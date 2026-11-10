// Chapter 05：声明 GGML Q3_K super-block 的 CPU 参考反量化接口。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chapter05_quantization {

constexpr std::size_t kQ3KBlockElements = 256;
constexpr std::size_t kQ3KBlockBytes = 110;

void decode_q3_k_block(const std::uint8_t* data, float* output);

bool dequantize_q3_k(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error = nullptr);

}  // namespace chapter05_quantization
