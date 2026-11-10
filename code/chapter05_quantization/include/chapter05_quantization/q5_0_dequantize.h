// Chapter 05：声明 GGML Q5_0 block 的 CPU 参考反量化接口。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chapter05_quantization {

constexpr std::size_t kQ5_0BlockElements = 32;
constexpr std::size_t kQ5_0BlockBytes = 22;

void decode_q5_0_block(const std::uint8_t* data, float* output);

bool dequantize_q5_0(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error = nullptr);

}  // namespace chapter05_quantization
