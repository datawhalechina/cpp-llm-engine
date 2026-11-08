// Chapter 05：声明 Q6_K Linear 的 reference、fused 和 backend 适配接口。
// backend 通过统一 LinearRequest 接收 packed 权重和 Float32 激活。
// 验证入口：chapter05_quantization_test。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cpp_llm_engine/core/linear_contract.h"

namespace chapter05_quantization {

class Q6KLinearReference final {
public:
    // 将 row-major Q6_K 权重解码到 float reference，再计算 W * input。
    static bool forward(const std::uint8_t* weights,
                        std::size_t weight_bytes,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error = nullptr);
};

class Q6KLinearFused final {
public:
    // 直接从 Q6_K block 读取并累加 dot product，不创建中间权重 buffer。
    static bool forward(const std::uint8_t* weights,
                        std::size_t weight_bytes,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error = nullptr);
};

class Q6KLinearFusedBackend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
    cpp_llm_engine::core::Status compute_batch(
        const cpp_llm_engine::core::LinearBatchRequest& request) const override;
};

const Q6KLinearFusedBackend& q6k_fused_backend();

class Q6KLinearAvx2 final {
public:
    // 在 Q6_K block 上直接执行 AVX2/FMA 解码与点积，不创建完整 float 权重。
    static bool forward(const std::uint8_t* weights,
                        std::size_t weight_bytes,
                        std::size_t rows,
                        std::size_t cols,
                        const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error = nullptr);
};

class Q6KLinearAvx2Backend final : public cpp_llm_engine::core::LinearBackend {
public:
    cpp_llm_engine::core::BackendKind kind() const override;
    std::string_view name() const override;
    cpp_llm_engine::core::Status compute(
        const cpp_llm_engine::core::LinearRequest& request) const override;
};

const Q6KLinearAvx2Backend& q6k_avx2_backend();

// AVX2 backend 还依赖 FMA；调用专用实现前先检查运行机器能力。
bool q6k_avx2_supported();

}  // namespace chapter05_quantization
