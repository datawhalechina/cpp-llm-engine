// 多模态 CUDA：单个 SigLIP Block 设备实现
#include "multimodal/cuda_siglip_block.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_linear_weight.h"
#include "gguf_tensor_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace multimodal {
namespace {

using cpp_llm_engine::core::DType;
using cpp_llm_engine::core::Status;
using cpp_llm_engine::cuda::CudaExecutionContext;
using cpp_llm_engine::cuda::CudaLinearWeight;
using cpp_llm_engine::cuda::CudaQuantizedMatrix;
using cpp_llm_engine::cuda::CudaTensor;

Status upload_vector(
    CudaExecutionContext* context,
    const std::vector<float>& host,
    CudaTensor* output) {
    Status status = CudaTensor::create_contiguous(
        {host.size()}, DType::Float32, output);
    if (status.is_ok()) {
        status = context->upload(
            output, host.data(), host.size() * sizeof(float));
    }
    return status;
}

Status create_tensor(
    const std::vector<std::size_t>& shape,
    CudaTensor* output) {
    return CudaTensor::create_contiguous(shape, DType::Float32, output);
}

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index,
    const std::string& name) {
    const auto iterator = std::find_if(
        index.tensors.begin(), index.tensors.end(),
        [&name](const chapter01::GgufTensorInfo& tensor) {
            return tensor.name == name;
        });
    return iterator == index.tensors.end() ? nullptr : &*iterator;
}

Status load_tensor(
    const std::string& path,
    const chapter01::GgufModelIndex& index,
    const std::string& name,
    std::size_t expected_elements,
    chapter01::Tensor* output) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) return Status::error("missing tensor: " + name);
    std::string error;
    if (!chapter01::GgufTensorLoader::load_tensor(
            path, index, *info, output, &error)) {
        return Status::error(error);
    }
    if (output->numel() != expected_elements) {
        return Status::error("tensor element count mismatch: " + name);
    }
    return Status::ok();
}

Status upload_f32_tensor(
    CudaExecutionContext* context,
    const std::string& path,
    const chapter01::GgufModelIndex& index,
    const std::string& name,
    std::size_t elements,
    CudaTensor* output) {
    chapter01::Tensor host;
    Status status = load_tensor(path, index, name, elements, &host);
    if (!status.is_ok()) return status;
    if (host.dtype() != chapter01::DType::Float32) {
        return Status::error("tensor dtype must be F32: " + name);
    }
    status = CudaTensor::create_contiguous({elements}, DType::Float32, output);
    if (status.is_ok()) {
        status = context->upload(output, host.raw_data(), host.bytes());
    }
    return status;
}

}  // namespace

Status CudaSiglipBlockWorkspace::create(
    std::size_t token_count,
    std::size_t hidden_size,
    std::size_t attention_heads,
    std::size_t feed_forward_size,
    CudaSiglipBlockWorkspace* output) {
    if (output == nullptr || token_count == 0 || hidden_size == 0 ||
        attention_heads == 0 || hidden_size % attention_heads != 0 ||
        feed_forward_size == 0) {
        return Status::error("CUDA SigLIP workspace shape is invalid");
    }
    const std::size_t head_dimension = hidden_size / attention_heads;
    CudaSiglipBlockWorkspace replacement;
    Status status = create_tensor(
        {token_count, hidden_size}, &replacement.normalized_attention);
    if (status.is_ok()) status = create_tensor(
        {token_count, attention_heads, head_dimension}, &replacement.query);
    if (status.is_ok()) status = create_tensor(
        {token_count, attention_heads, head_dimension}, &replacement.key);
    if (status.is_ok()) status = create_tensor(
        {token_count, attention_heads, head_dimension}, &replacement.value);
    if (status.is_ok()) status = create_tensor(
        {token_count, attention_heads, head_dimension},
        &replacement.attention_context);
    if (status.is_ok()) status = create_tensor(
        {token_count, hidden_size}, &replacement.attention_projection);
    if (status.is_ok()) status = create_tensor(
        {token_count, hidden_size}, &replacement.attention_residual);
    if (status.is_ok()) status = create_tensor(
        {token_count, hidden_size}, &replacement.normalized_feed_forward);
    if (status.is_ok()) status = create_tensor(
        {token_count, feed_forward_size}, &replacement.feed_forward_hidden);
    if (status.is_ok()) status = create_tensor(
        {token_count, feed_forward_size}, &replacement.feed_forward_activated);
    if (status.is_ok()) status = create_tensor(
        {token_count, hidden_size}, &replacement.feed_forward_output);
    if (!status.is_ok()) return status;
    replacement.token_count_ = token_count;
    replacement.hidden_size_ = hidden_size;
    replacement.attention_heads_ = attention_heads;
    replacement.feed_forward_size_ = feed_forward_size;
    *output = std::move(replacement);
    return Status::ok();
}

bool CudaSiglipBlockWorkspace::valid_for(
    std::size_t token_count,
    std::size_t hidden_size,
    std::size_t attention_heads,
    std::size_t feed_forward_size) const {
    return token_count_ == token_count && hidden_size_ == hidden_size &&
        attention_heads_ == attention_heads &&
        feed_forward_size_ == feed_forward_size;
}

std::size_t CudaSiglipBlockWorkspace::token_count() const {
    return token_count_;
}

Status CudaSiglipBlock::upload_gguf_linear(
    CudaExecutionContext* context,
    const std::string& path,
    const chapter01::GgufModelIndex& index,
    const std::string& weight_name,
    const std::string& bias_name,
    std::size_t input_size,
    std::size_t output_size,
    CudaSiglipBlock::DeviceLinear* output) {
    if (input_size != 0 &&
        output_size > std::numeric_limits<std::size_t>::max() / input_size) {
        return Status::error("linear weight shape overflows: " + weight_name);
    }
    const std::size_t elements = input_size * output_size;
    chapter01::Tensor host_weight;
    Status status = load_tensor(
        path, index, weight_name, elements, &host_weight);
    if (!status.is_ok()) return status;

    CudaSiglipBlock::DeviceLinear replacement;
    if (host_weight.dtype() == chapter01::DType::Float32) {
        const std::vector<float> source = host_weight.to_vector<float>();
        std::vector<float> transposed(elements);
        for (std::size_t destination = 0; destination < output_size;
             ++destination) {
            for (std::size_t source_index = 0; source_index < input_size;
                 ++source_index) {
                transposed[source_index * output_size + destination] =
                    source[destination * input_size + source_index];
            }
        }
        status = create_tensor(
            {input_size, output_size}, &replacement.weight);
        if (status.is_ok()) {
            status = context->upload(
                &replacement.weight, transposed.data(),
                transposed.size() * sizeof(float));
        }
    } else if (host_weight.dtype() == chapter01::DType::Q8_0) {
        status = CudaQuantizedMatrix::create(
            output_size, input_size,
            cpp_llm_engine::core::WeightFormat::Q8_0,
            &replacement.quantized_weight);
        if (status.is_ok()) {
            status = context->upload(
                &replacement.quantized_weight,
                host_weight.raw_data(), host_weight.bytes());
        }
        replacement.quantized = status.is_ok();
    } else {
        return Status::error("linear tensor dtype must be F32 or Q8_0: " + weight_name);
    }
    if (status.is_ok()) {
        status = upload_f32_tensor(
            context, path, index, bias_name, output_size, &replacement.bias);
    }
    if (!status.is_ok()) return status;
    replacement.input_size = input_size;
    replacement.output_size = output_size;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaSiglipBlock::upload_linear(
    CudaExecutionContext* context,
    const LinearWeights& host,
    DeviceLinear* output) {
    if (context == nullptr || output == nullptr || !host.valid()) {
        return Status::error("CUDA SigLIP Linear upload input is invalid");
    }
    std::vector<float> transposed(host.input_size * host.output_size);
    for (std::size_t destination = 0; destination < host.output_size;
         ++destination) {
        for (std::size_t source = 0; source < host.input_size; ++source) {
            transposed[source * host.output_size + destination] =
                host.weight[destination * host.input_size + source];
        }
    }
    std::vector<float> bias = host.bias;
    if (bias.empty()) bias.assign(host.output_size, 0.0F);

    DeviceLinear replacement;
    Status status = create_tensor(
        {host.input_size, host.output_size}, &replacement.weight);
    if (status.is_ok()) {
        status = context->upload(
            &replacement.weight, transposed.data(),
            transposed.size() * sizeof(float));
    }
    if (status.is_ok()) status = upload_vector(context, bias, &replacement.bias);
    if (!status.is_ok()) return status;
    replacement.input_size = host.input_size;
    replacement.output_size = host.output_size;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaSiglipBlock::run_linear(
    CudaExecutionContext* context,
    const CudaTensor& input,
    const DeviceLinear& linear,
    CudaTensor* output) {
    const CudaLinearWeight weight = linear.quantized
        ? CudaLinearWeight::quantized(&linear.quantized_weight)
        : CudaLinearWeight::float32(&linear.weight);
    Status status = context->linear_batch(input, weight, output);
    if (status.is_ok()) {
        status = context->add_bias_batch(*output, linear.bias, output);
    }
    return status;
}

Status CudaSiglipBlock::upload(
    CudaExecutionContext* context,
    const SiglipBlockWeights& host,
    std::size_t attention_heads,
    float layer_norm_epsilon,
    CudaSiglipBlock* output) {
    const std::size_t hidden_size = host.query.input_size;
    if (context == nullptr || output == nullptr || !context->valid() ||
        hidden_size == 0 || !host.valid(hidden_size) || attention_heads == 0 ||
        hidden_size % attention_heads != 0 ||
        !std::isfinite(layer_norm_epsilon) || layer_norm_epsilon <= 0.0F) {
        return Status::error("CUDA SigLIP Block upload input is invalid");
    }

    CudaSiglipBlock replacement;
    Status status = upload_vector(
        context, host.layer_norm_1.weight, &replacement.norm_1_weight_);
    if (status.is_ok()) status = upload_vector(
        context, host.layer_norm_1.bias, &replacement.norm_1_bias_);
    if (status.is_ok()) status = upload_linear(context, host.query, &replacement.query_);
    if (status.is_ok()) status = upload_linear(context, host.key, &replacement.key_);
    if (status.is_ok()) status = upload_linear(context, host.value, &replacement.value_);
    if (status.is_ok()) status = upload_linear(
        context, host.attention_output, &replacement.attention_output_);
    if (status.is_ok()) status = upload_vector(
        context, host.layer_norm_2.weight, &replacement.norm_2_weight_);
    if (status.is_ok()) status = upload_vector(
        context, host.layer_norm_2.bias, &replacement.norm_2_bias_);
    if (status.is_ok()) status = upload_linear(
        context, host.feed_forward_input, &replacement.feed_forward_input_);
    if (status.is_ok()) status = upload_linear(
        context, host.feed_forward_output, &replacement.feed_forward_output_);
    if (!status.is_ok()) return status;

    replacement.hidden_size_ = hidden_size;
    replacement.attention_heads_ = attention_heads;
    replacement.feed_forward_size_ = host.feed_forward_input.output_size;
    replacement.layer_norm_epsilon_ = layer_norm_epsilon;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaSiglipBlock::upload_gguf(
    CudaExecutionContext* context,
    const std::string& path,
    const chapter01::GgufModelIndex& index,
    const VisionModelConfig& config,
    const VisionTensorLayout& layout,
    std::size_t block_index,
    CudaSiglipBlock* output) {
    if (context == nullptr || output == nullptr || !context->valid() ||
        block_index >= layout.blocks.size() || config.hidden_size == 0 ||
        config.intermediate_size == 0 || config.attention_head_count == 0 ||
        config.hidden_size % config.attention_head_count != 0 ||
        !std::isfinite(config.layer_norm_epsilon) ||
        config.layer_norm_epsilon <= 0.0F) {
        return Status::error("CUDA SigLIP GGUF upload input is invalid");
    }
    const VisionBlockLayout& block = layout.blocks[block_index];
    CudaSiglipBlock replacement;
    Status status = upload_f32_tensor(
        context, path, index, block.layer_norm_1_weight,
        config.hidden_size, &replacement.norm_1_weight_);
    if (status.is_ok()) status = upload_f32_tensor(
        context, path, index, block.layer_norm_1_bias,
        config.hidden_size, &replacement.norm_1_bias_);
    if (status.is_ok()) status = upload_gguf_linear(
        context, path, index, block.attention_query_weight,
        block.attention_query_bias, config.hidden_size, config.hidden_size,
        &replacement.query_);
    if (status.is_ok()) status = upload_gguf_linear(
        context, path, index, block.attention_key_weight,
        block.attention_key_bias, config.hidden_size, config.hidden_size,
        &replacement.key_);
    if (status.is_ok()) status = upload_gguf_linear(
        context, path, index, block.attention_value_weight,
        block.attention_value_bias, config.hidden_size, config.hidden_size,
        &replacement.value_);
    if (status.is_ok()) status = upload_gguf_linear(
        context, path, index, block.attention_output_weight,
        block.attention_output_bias, config.hidden_size, config.hidden_size,
        &replacement.attention_output_);
    if (status.is_ok()) status = upload_f32_tensor(
        context, path, index, block.layer_norm_2_weight,
        config.hidden_size, &replacement.norm_2_weight_);
    if (status.is_ok()) status = upload_f32_tensor(
        context, path, index, block.layer_norm_2_bias,
        config.hidden_size, &replacement.norm_2_bias_);
    if (status.is_ok()) status = upload_gguf_linear(
        context, path, index, block.feed_forward_down_weight,
        block.feed_forward_down_bias, config.hidden_size,
        config.intermediate_size, &replacement.feed_forward_input_);
    if (status.is_ok()) status = upload_gguf_linear(
        context, path, index, block.feed_forward_up_weight,
        block.feed_forward_up_bias, config.intermediate_size,
        config.hidden_size, &replacement.feed_forward_output_);
    if (!status.is_ok()) return status;

    replacement.hidden_size_ = config.hidden_size;
    replacement.attention_heads_ = config.attention_head_count;
    replacement.feed_forward_size_ = config.intermediate_size;
    replacement.layer_norm_epsilon_ = config.layer_norm_epsilon;
    *output = std::move(replacement);
    return Status::ok();
}

Status CudaSiglipBlock::forward(
    CudaExecutionContext* context,
    const CudaTensor& input,
    CudaTensor* output) const {
    if (context == nullptr || output == nullptr || !context->valid() ||
        !valid()) {
        return Status::error("CUDA SigLIP Block forward input is invalid");
    }
    const auto& shape = input.contract().shape.dimensions;
    if (shape.size() != 2 || shape[1] != hidden_size_) {
        return Status::error("CUDA SigLIP Block input shape is invalid");
    }
    CudaSiglipBlockWorkspace workspace;
    Status status = CudaSiglipBlockWorkspace::create(
        shape[0], hidden_size_, attention_heads_, feed_forward_size_,
        &workspace);
    if (status.is_ok()) {
        status = forward_with_workspace(context, input, &workspace, output);
    }
    // The one-shot wrapper must keep its local workspace alive until work ends.
    if (status.is_ok()) status = context->synchronize();
    return status;
}

Status CudaSiglipBlock::forward_with_workspace(
    CudaExecutionContext* context,
    const CudaTensor& input,
    CudaSiglipBlockWorkspace* workspace,
    CudaTensor* output) const {
    return forward_batch_with_workspace(
        context, input, 1, workspace, output);
}

Status CudaSiglipBlock::forward_batch_with_workspace(
    CudaExecutionContext* context,
    const CudaTensor& input,
    std::size_t batch_size,
    CudaSiglipBlockWorkspace* workspace,
    CudaTensor* output) const {
    if (context == nullptr || workspace == nullptr || output == nullptr ||
        !context->valid() || !valid() || batch_size == 0) {
        return Status::error("CUDA SigLIP Block workspace input is invalid");
    }
    const auto& shape = input.contract().shape.dimensions;
    if (shape.size() != 2 || shape[0] % batch_size != 0 ||
        shape[1] != hidden_size_ ||
        output->contract().shape.dimensions != shape ||
        !workspace->valid_for(
            shape[0], hidden_size_, attention_heads_, feed_forward_size_)) {
        return Status::error("CUDA SigLIP Block workspace shape is invalid");
    }
    Status status = context->layer_norm_batch(
        input, norm_1_weight_, norm_1_bias_, &workspace->normalized_attention,
        layer_norm_epsilon_);
    if (status.is_ok()) status = run_linear(
        context, workspace->normalized_attention, query_, &workspace->query);
    if (status.is_ok()) status = run_linear(
        context, workspace->normalized_attention, key_, &workspace->key);
    if (status.is_ok()) status = run_linear(
        context, workspace->normalized_attention, value_, &workspace->value);
    if (status.is_ok()) status = context->vision_attention_batch(
        workspace->query, workspace->key, workspace->value, batch_size,
        shape[0] / batch_size, &workspace->attention_context);
    if (status.is_ok()) status = run_linear(
        context, workspace->attention_context, attention_output_,
        &workspace->attention_projection);
    if (status.is_ok()) status = context->vector_add(
        input, workspace->attention_projection, &workspace->attention_residual);
    if (status.is_ok()) status = context->layer_norm_batch(
        workspace->attention_residual, norm_2_weight_, norm_2_bias_,
        &workspace->normalized_feed_forward, layer_norm_epsilon_);
    if (status.is_ok()) status = run_linear(
        context, workspace->normalized_feed_forward, feed_forward_input_,
        &workspace->feed_forward_hidden);
    if (status.is_ok()) status = context->gelu(
        workspace->feed_forward_hidden, &workspace->feed_forward_activated);
    if (status.is_ok()) status = run_linear(
        context, workspace->feed_forward_activated, feed_forward_output_,
        &workspace->feed_forward_output);
    if (status.is_ok()) status = context->vector_add(
        workspace->attention_residual, workspace->feed_forward_output, output);
    return status;
}

bool CudaSiglipBlock::valid() const {
    return hidden_size_ > 0 && attention_heads_ > 0 &&
        hidden_size_ % attention_heads_ == 0 && feed_forward_size_ > 0 &&
        std::isfinite(layer_norm_epsilon_) && layer_norm_epsilon_ > 0.0F;
}

std::size_t CudaSiglipBlock::hidden_size() const { return hidden_size_; }
std::size_t CudaSiglipBlock::attention_heads() const {
    return attention_heads_;
}

}  // namespace multimodal
