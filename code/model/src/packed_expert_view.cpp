// Model：实现 GGUF 三维 packed expert 的连续切片校验。
// 职责：校验 dtype、shape、block 字节数并计算 expert 字节窗口。
// 契约：失败保持调用方输出不变，不复制权重 payload。
// 验证：packed_expert_view_test。
#include "cpp_llm_engine/model/packed_expert_view.h"

#include <limits>

namespace cpp_llm_engine::model {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

bool format_of(chapter01::DType dtype,
               cpp_llm_engine::core::WeightFormat* format,
               std::size_t* block_elements,
               std::size_t* block_bytes) {
    using chapter01::DType;
    using cpp_llm_engine::core::WeightFormat;
    if (dtype == DType::Q3_K) {
        *format = WeightFormat::Q3K;
        *block_elements = 256;
        *block_bytes = 110;
        return true;
    }
    if (dtype == DType::Q4_K) {
        *format = WeightFormat::Q4K;
        *block_elements = 256;
        *block_bytes = 144;
        return true;
    }
    if (dtype == DType::Q5_0) {
        *format = WeightFormat::Q5_0;
        *block_elements = 32;
        *block_bytes = 22;
        return true;
    }
    if (dtype == DType::Q5_1) {
        *format = WeightFormat::Q5_1;
        *block_elements = 32;
        *block_bytes = 24;
        return true;
    }
    if (dtype == DType::Q5_K) {
        *format = WeightFormat::Q5K;
        *block_elements = 256;
        *block_bytes = 176;
        return true;
    }
    if (dtype == DType::Q6_K) {
        *format = WeightFormat::Q6K;
        *block_elements = 256;
        *block_bytes = 210;
        return true;
    }
    if (dtype == DType::Q8_0) {
        *format = WeightFormat::Q8_0;
        *block_elements = 32;
        *block_bytes = 34;
        return true;
    }
    return false;
}

}  // namespace

bool PackedExpertView::create(const chapter01::Tensor& packed,
                              std::size_t expert_index,
                              PackedExpertView* output,
                              std::string* error) {
    if (output == nullptr) {
        return fail(error, "packed expert output must not be null");
    }
    const auto& shape = packed.shape();
    if (shape.size() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
        return fail(error, "packed expert tensor must have positive [cols, rows, experts] shape");
    }
    const auto cols = static_cast<std::size_t>(shape[0]);
    const auto rows = static_cast<std::size_t>(shape[1]);
    const auto experts = static_cast<std::size_t>(shape[2]);
    if (expert_index >= experts) {
        return fail(error, "expert index is outside packed tensor");
    }

    PackedExpertView view;
    std::size_t block_elements = 0;
    std::size_t block_bytes = 0;
    if (!format_of(packed.dtype(), &view.format, &block_elements, &block_bytes)) {
        return fail(error, "packed expert dtype is not supported");
    }
    if (cols % block_elements != 0 || rows > std::numeric_limits<std::size_t>::max() / (cols / block_elements)) {
        return fail(error, "packed expert matrix is not block aligned");
    }
    const std::size_t blocks = rows * (cols / block_elements);
    if (blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
        return fail(error, "packed expert byte count overflows host size");
    }
    view.bytes = blocks * block_bytes;
    if (experts > std::numeric_limits<std::size_t>::max() / view.bytes ||
        packed.bytes() != experts * view.bytes || packed.raw_data() == nullptr) {
        return fail(error, "packed expert payload does not match shape");
    }
    view.data = reinterpret_cast<const std::uint8_t*>(packed.raw_data()) + expert_index * view.bytes;
    view.rows = rows;
    view.cols = cols;
    *output = view;
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace cpp_llm_engine::model
