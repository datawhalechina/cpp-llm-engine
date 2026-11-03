// Chapter 05：声明 GGML Q8_0 block 的 CPU reference 解码接口。
// 契约：每个 block 保存 32 个值，由一个 FP16 scale 和 32 个 int8 quant 组成。
// 验证：chapter05_quantization_test 与真实 SmolVLM mmproj。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chapter05_quantization {

constexpr std::size_t kQ8_0BlockElements = 32;
constexpr std::size_t kQ8_0BlockBytes = 34;

bool dequantize_q8_0(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error = nullptr);

}  // namespace chapter05_quantization
