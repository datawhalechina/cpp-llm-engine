// Chapter 05：声明 GGML Q4_K super-block 的 CPU reference 解码接口。
// 验证：构造 block 与真实 TinyLlama payload 均和 llama.cpp 对照。
#pragma once

#include "cpp_llm_engine/core/quantization_contract.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chapter05_quantization {

constexpr std::size_t kQ4KBlockElements =
    cpp_llm_engine::core::kQuantizedBlockElements;
constexpr std::size_t kQ4KBlockBytes = cpp_llm_engine::core::kQ4KBlockBytes;

// Decodes llama.cpp-compatible Q4_K blocks into a CPU float reference buffer.
bool dequantize_q4_k(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error = nullptr);

}  // namespace chapter05_quantization
