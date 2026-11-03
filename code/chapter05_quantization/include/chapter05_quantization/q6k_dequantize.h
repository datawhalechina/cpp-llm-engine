// 中文：声明 Q6_K block 的 CPU reference 反量化接口；输入输出契约由 Chapter05 测试验证。
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
