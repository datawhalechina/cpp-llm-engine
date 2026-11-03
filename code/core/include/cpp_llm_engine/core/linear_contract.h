// Core：Linear backend 请求契约
// 职责：固定 row-major weights[rows, cols] * input[cols] 的输入输出布局。
// 契约：weight_format 标识 Float32、Q4K、Q6K 或 Q8_0，backend 写入 output。
// 验证：Chapter 05 quantized backend 测试。
#pragma once

#include "backend.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cpp_llm_engine::core {

enum class WeightFormat : std::uint8_t {
    Unknown = 0,
    Float32,
    Q4K,
    Q6K,
    Q8_0,
};

// Row-major contract: weights[rows, cols] * input[cols] = output[rows].
struct LinearRequest final {
    const void* weights = nullptr;
    std::size_t weight_bytes = 0;
    const float* input = nullptr;
    std::size_t input_size = 0;
    float* output = nullptr;
    std::size_t output_size = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    WeightFormat weight_format = WeightFormat::Unknown;
};

// Row-major contract: input[batch, cols] * weights[rows, cols]^T
//                   = output[batch, rows].
struct LinearBatchRequest final {
    const void* weights = nullptr;
    std::size_t weight_bytes = 0;
    const float* input = nullptr;
    std::size_t input_size = 0;
    float* output = nullptr;
    std::size_t output_size = 0;
    std::size_t batch_size = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    WeightFormat weight_format = WeightFormat::Unknown;
};

class LinearBackend : public Backend {
public:
    ~LinearBackend() override = default;
    virtual Status compute(const LinearRequest& request) const = 0;
    virtual Status compute_batch(const LinearBatchRequest& request) const {
        if (request.batch_size == 0 || request.cols == 0 || request.rows == 0 ||
            request.batch_size > std::numeric_limits<std::size_t>::max() / request.cols ||
            request.batch_size > std::numeric_limits<std::size_t>::max() / request.rows ||
            request.input_size != request.batch_size * request.cols ||
            request.output_size < request.batch_size * request.rows) {
            return Status::error("Linear batch buffer sizes do not match shape");
        }
        for (std::size_t batch = 0; batch < request.batch_size; ++batch) {
            const LinearRequest row{
                request.weights,
                request.weight_bytes,
                request.input + batch * request.cols,
                request.cols,
                request.output + batch * request.rows,
                request.rows,
                request.rows,
                request.cols,
                request.weight_format,
            };
            const auto status = compute(row);
            if (!status.is_ok()) return status;
        }
        return Status::ok();
    }
};

}  // namespace cpp_llm_engine::core
