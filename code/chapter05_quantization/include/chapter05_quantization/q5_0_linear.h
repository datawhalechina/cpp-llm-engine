// Chapter 05：声明 Q5_0 Linear 的参考路径与边解码边点积 CPU backend。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

#include <string_view>

namespace chapter05_quantization {

class Q5_0LinearFusedBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
};

const Q5_0LinearFusedBackend& q5_0_fused_backend();

}  // namespace chapter05_quantization
