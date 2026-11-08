// Chapter 05：声明 Q6_K block 的 CPU reference 反量化接口。
// 输入必须符合 GGUF Q6_K block 大小，输出按 Float32 元素展开。
// 验证入口：chapter05_quantization_test。
#pragma once

#include "cpp_llm_engine/core/quantization_contract.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chapter05_quantization {

constexpr std::size_t kQ6KBlockElements =
    cpp_llm_engine::core::kQuantizedBlockElements;
constexpr std::size_t kQ6KBlockBytes = cpp_llm_engine::core::kQ6KBlockBytes;

// Decodes llama.cpp-compatible Q6_K blocks into a CPU float reference buffer.
bool dequantize_q6_k(const std::uint8_t* data,
                     std::size_t byte_count,
                     std::size_t element_count,
                     std::vector<float>* output,
                     std::string* error = nullptr);

}  // namespace chapter05_quantization
