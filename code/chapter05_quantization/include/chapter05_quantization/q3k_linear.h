// Chapter 05：声明 Q3_K Linear 的参考路径与边解码边点积 CPU backend。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace chapter05_quantization {

class Q3KLinearReference final {
public:
    static bool forward(const std::uint8_t* weights, std::size_t weight_bytes,
                        std::size_t rows, std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output, std::string* error = nullptr);
};

class Q3KLinearFusedBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
    cpp_llm_engine::core::Status compute_batch(
        const cpp_llm_engine::core::LinearBatchRequest& request) const override;
};

const Q3KLinearFusedBackend& q3k_fused_backend();

}  // namespace chapter05_quantization
