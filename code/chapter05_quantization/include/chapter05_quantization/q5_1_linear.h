// Chapter 05：声明 Q5_1 Linear 的参考路径与边解码边点积 CPU backend。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace chapter05_quantization {

class Q5_1LinearReference final {
public:
    static bool forward(const std::uint8_t* weights, std::size_t weight_bytes,
                        std::size_t rows, std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output, std::string* error = nullptr);
};

class Q5_1LinearFusedBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
};

const Q5_1LinearFusedBackend& q5_1_fused_backend();

}  // namespace chapter05_quantization
