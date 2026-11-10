// Chapter 05：声明 IQ4_NL 边解码边点积的 CPU Linear backend。
#pragma once

#include "cpp_llm_engine/core/linear_contract.h"

#include <string_view>

namespace chapter05_quantization {

class IQ4NLLinearBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
};

const IQ4NLLinearBackend& iq4_nl_backend();

}  // namespace chapter05_quantization
