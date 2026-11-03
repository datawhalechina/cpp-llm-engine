// Core：非拥有 Tensor 数据引用
// 职责：把 Storage 基址与 TensorContract 组合成只读或可写 View。
// 契约：不延长 Storage 生命周期；调用方保证引用使用期间数据仍然有效。
// 验证：code/core/tests/core_contract_test.cpp。
#pragma once

#include "tensor_contract.h"

#include <cstddef>

namespace cpp_llm_engine::core {

struct TensorViewRef final {
    const void* storage_data = nullptr;
    TensorContract contract;

    Status validate(std::size_t element_size_bytes) const {
        if (storage_data == nullptr) {
            return Status::error("tensor view storage data must not be null");
        }
        return contract.validate_element_view(element_size_bytes);
    }
};

struct MutableTensorViewRef final {
    void* storage_data = nullptr;
    TensorContract contract;

    Status validate(std::size_t element_size_bytes) const {
        if (storage_data == nullptr) {
            return Status::error("mutable tensor view storage data must not be null");
        }
        return contract.validate_element_view(element_size_bytes);
    }

    TensorViewRef as_const() const {
        return TensorViewRef{storage_data, contract};
    }
};

}  // namespace cpp_llm_engine::core
