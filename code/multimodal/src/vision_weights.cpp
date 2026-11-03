// 多模态 CPU：视觉 GGUF 权重适配实现
// 职责：校验 tensor 名称、类型与元素数，解码 Q8_0，并转换为视觉数学接口。
// 契约：Linear 权重保持 GGUF/GGML 的 output-major 连续布局。
// 验证：multimodal_vision_weights_test 与真实 SmolVLM 首 Block。
#include "multimodal/vision_weights.h"

#include "chapter05_quantization/q8_0_dequantize.h"
#include "gguf_tensor_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace multimodal {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

bool checked_product(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) return false;
    *result = lhs * rhs;
    return true;
}

const chapter01::GgufTensorInfo* find_tensor(const chapter01::GgufModelIndex& index,
                                             const std::string& name) {
    const auto iterator = std::find_if(
        index.tensors.begin(), index.tensors.end(),
        [&name](const chapter01::GgufTensorInfo& tensor) { return tensor.name == name; });
    return iterator == index.tensors.end() ? nullptr : &*iterator;
}

bool load_values(const std::string& path,
                 const chapter01::GgufModelIndex& index,
                 const std::string& name,
                 std::size_t expected_elements,
                 bool allow_q8_0,
                 std::vector<float>* values,
                 std::string* error) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) return fail(error, "missing tensor: " + name);
    chapter01::Tensor tensor;
    if (!chapter01::GgufTensorLoader::load_tensor(path, index, *info, &tensor, error)) return false;
    if (tensor.numel() != expected_elements) {
        return fail(error, "tensor element count mismatch: " + name);
    }
    if (tensor.dtype() == chapter01::DType::Float32) {
        *values = tensor.to_vector<float>();
        return true;
    }
    if (allow_q8_0 && tensor.dtype() == chapter01::DType::Q8_0) {
        return chapter05_quantization::dequantize_q8_0(
            reinterpret_cast<const std::uint8_t*>(tensor.raw_data()),
            tensor.bytes(), tensor.numel(), values, error);
    }
    return fail(error, "tensor dtype must be F32" +
                           std::string(allow_q8_0 ? " or Q8_0: " : ": ") + name);
}

bool load_linear(const std::string& path,
                 const chapter01::GgufModelIndex& index,
                 const std::string& weight_name,
                 const std::string& bias_name,
                 std::size_t input_size,
                 std::size_t output_size,
                 LinearWeights* weights,
                 std::string* error) {
    std::size_t elements = 0;
    if (!checked_product(input_size, output_size, &elements)) {
        return fail(error, "linear weight shape overflows: " + weight_name);
    }
    LinearWeights loaded;
    loaded.input_size = input_size;
    loaded.output_size = output_size;
    if (!load_values(path, index, weight_name, elements, true, &loaded.weight, error) ||
        !load_values(path, index, bias_name, output_size, false, &loaded.bias, error)) {
        return false;
    }
    *weights = std::move(loaded);
    return true;
}

bool load_norm(const std::string& path,
               const chapter01::GgufModelIndex& index,
               const std::string& weight_name,
               const std::string& bias_name,
               std::size_t hidden_size,
               LayerNormWeights* weights,
               std::string* error) {
    LayerNormWeights loaded;
    if (!load_values(path, index, weight_name, hidden_size, false, &loaded.weight, error) ||
        !load_values(path, index, bias_name, hidden_size, false, &loaded.bias, error)) {
        return false;
    }
    *weights = std::move(loaded);
    return true;
}

}  // namespace

bool load_patch_embedding_weights(const std::string& path,
                                  const chapter01::GgufModelIndex& index,
                                  const VisionModelConfig& config,
                                  const VisionTensorLayout& layout,
                                  PatchEmbeddingWeights* weights,
                                  std::string* error) {
    if (weights == nullptr) return fail(error, "patch embedding weights output must not be null");
    std::size_t patch_area = 0;
    std::size_t patch_values = 0;
    const std::size_t patches_per_side =
        config.patch_size == 0 ? 0 : config.image_size / config.patch_size;
    std::size_t position_count = 0;
    if (!checked_product(config.patch_size, config.patch_size, &patch_area) ||
        !checked_product(3, patch_area, &patch_values) ||
        !checked_product(patches_per_side, patches_per_side, &position_count)) {
        return fail(error, "patch embedding shape overflows");
    }
    std::size_t patch_weight_elements = 0;
    std::size_t position_elements = 0;
    if (!checked_product(patch_values, config.hidden_size, &patch_weight_elements) ||
        !checked_product(position_count, config.hidden_size, &position_elements)) {
        return fail(error, "patch embedding payload shape overflows");
    }

    PatchEmbeddingWeights loaded;
    loaded.patch_values = patch_values;
    loaded.hidden_size = config.hidden_size;
    loaded.position_count = position_count;
    if (!load_values(path, index, layout.patch_embedding_weight,
                     patch_weight_elements, false, &loaded.patch_weight, error) ||
        !load_values(path, index, layout.patch_embedding_bias,
                     config.hidden_size, false, &loaded.patch_bias, error) ||
        !load_values(path, index, layout.position_embedding_weight,
                     position_elements, false, &loaded.position_weight, error)) {
        return false;
    }
    *weights = std::move(loaded);
    if (error != nullptr) error->clear();
    return true;
}

bool load_siglip_block_weights(const std::string& path,
                               const chapter01::GgufModelIndex& index,
                               const VisionModelConfig& config,
                               const VisionTensorLayout& layout,
                               std::size_t block_index,
                               SiglipBlockWeights* weights,
                               std::string* error) {
    if (weights == nullptr) return fail(error, "SigLIP block weights output must not be null");
    if (block_index >= layout.blocks.size()) return fail(error, "SigLIP block index is out of range");
    const VisionBlockLayout& block = layout.blocks[block_index];
    SiglipBlockWeights loaded;
    if (!load_norm(path, index, block.layer_norm_1_weight, block.layer_norm_1_bias,
                   config.hidden_size, &loaded.layer_norm_1, error) ||
        !load_linear(path, index, block.attention_query_weight, block.attention_query_bias,
                     config.hidden_size, config.hidden_size, &loaded.query, error) ||
        !load_linear(path, index, block.attention_key_weight, block.attention_key_bias,
                     config.hidden_size, config.hidden_size, &loaded.key, error) ||
        !load_linear(path, index, block.attention_value_weight, block.attention_value_bias,
                     config.hidden_size, config.hidden_size, &loaded.value, error) ||
        !load_linear(path, index, block.attention_output_weight, block.attention_output_bias,
                     config.hidden_size, config.hidden_size, &loaded.attention_output, error) ||
        !load_norm(path, index, block.layer_norm_2_weight, block.layer_norm_2_bias,
                   config.hidden_size, &loaded.layer_norm_2, error) ||
        !load_linear(path, index, block.feed_forward_down_weight, block.feed_forward_down_bias,
                     config.hidden_size, config.intermediate_size,
                     &loaded.feed_forward_input, error) ||
        !load_linear(path, index, block.feed_forward_up_weight, block.feed_forward_up_bias,
                     config.intermediate_size, config.hidden_size,
                     &loaded.feed_forward_output, error)) {
        return false;
    }
    if (!loaded.valid(config.hidden_size)) return fail(error, "loaded SigLIP block weights are invalid");
    *weights = std::move(loaded);
    if (error != nullptr) error->clear();
    return true;
}

bool load_post_layer_norm_weights(const std::string& path,
                                  const chapter01::GgufModelIndex& index,
                                  const VisionModelConfig& config,
                                  const VisionTensorLayout& layout,
                                  LayerNormWeights* weights,
                                  std::string* error) {
    if (weights == nullptr) return fail(error, "post layer norm weights output must not be null");
    LayerNormWeights loaded;
    if (!load_norm(path, index, layout.post_layer_norm_weight,
                   layout.post_layer_norm_bias, config.hidden_size, &loaded, error)) {
        return false;
    }
    *weights = std::move(loaded);
    if (error != nullptr) error->clear();
    return true;
}

bool load_idefics3_projector_weights(const std::string& path,
                                     const chapter01::GgufModelIndex& index,
                                     const VisionModelConfig& config,
                                     const VisionTensorLayout& layout,
                                     LinearWeights* weights,
                                     std::string* error) {
    if (weights == nullptr) return fail(error, "IDEFICS3 projector weights output must not be null");
    std::size_t merged_hidden = 0;
    std::size_t elements = 0;
    if (!checked_product(config.merge_size, config.merge_size, &merged_hidden) ||
        !checked_product(merged_hidden, config.hidden_size, &merged_hidden) ||
        !checked_product(merged_hidden, config.projection_dim, &elements)) {
        return fail(error, "IDEFICS3 projector weight shape overflows");
    }
    LinearWeights loaded;
    loaded.input_size = merged_hidden;
    loaded.output_size = config.projection_dim;
    if (!load_values(path, index, layout.projector_weight,
                     elements, true, &loaded.weight, error)) {
        return false;
    }
    if (!loaded.valid()) return fail(error, "loaded IDEFICS3 projector weights are invalid");
    *weights = std::move(loaded);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace multimodal
