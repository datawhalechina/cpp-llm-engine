// DeepSeek2 CUDA Model：加载并执行一个稠密层和后续 MoE 层。
#include "cpp_llm_engine/model/real_deepseek2_cuda_model.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_quantized_matrix.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer.h"
#include "cpp_llm_engine/model/cuda_deepseek2_dense_layer_weights.h"
#include "cpp_llm_engine/model/cuda_deepseek2_moe_feed_forward.h"
#include "cpp_llm_engine/model/cuda_deepseek2_moe_layer.h"
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>

namespace cpp_llm_engine::model {
namespace {

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index, const std::string& name) {
    const auto found = std::find_if(
        index.tensors.begin(), index.tensors.end(),
        [&name](const chapter01::GgufTensorInfo& tensor) {
            return tensor.name == name;
        });
    return found == index.tensors.end() ? nullptr : &*found;
}

bool load_named(const std::string& model_path,
                const chapter01::GgufModelIndex& index,
                const std::string& name,
                chapter01::Tensor* output,
                std::string* error) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) {
        if (error != nullptr) *error = "DeepSeek2 CUDA tensor is missing: " + name;
        return false;
    }
    return chapter01::GgufTensorLoader::load_tensor(
        model_path, index, *info, output, error);
}

core::WeightFormat format_of(chapter01::DType dtype) {
    if (dtype == chapter01::DType::Q2_K) return core::WeightFormat::Q2K;
    if (dtype == chapter01::DType::Q3_K) return core::WeightFormat::Q3K;
    if (dtype == chapter01::DType::Q4_K) return core::WeightFormat::Q4K;
    if (dtype == chapter01::DType::Q6_K) return core::WeightFormat::Q6K;
    if (dtype == chapter01::DType::IQ4_NL) return core::WeightFormat::IQ4_NL;
    return core::WeightFormat::Unknown;
}

core::Status upload_matrix(cuda::CudaExecutionContext* context,
                           const chapter01::Tensor& host,
                           cuda::CudaQuantizedMatrix* output) {
    if (host.shape().size() != 2 || host.shape()[0] <= 0 ||
        host.shape()[1] <= 0 || host.raw_data() == nullptr) {
        return core::Status::error("DeepSeek2 CUDA matrix is invalid");
    }
    const core::WeightFormat format = format_of(host.dtype());
    if (format == core::WeightFormat::Unknown) {
        return core::Status::error("DeepSeek2 CUDA matrix format is unsupported");
    }
    cuda::CudaQuantizedMatrix replacement;
    core::Status status = cuda::CudaQuantizedMatrix::create(
        static_cast<std::size_t>(host.shape()[1]),
        static_cast<std::size_t>(host.shape()[0]), format, &replacement);
    if (status.is_ok()) status = context->upload(
        &replacement, host.raw_data(), host.bytes());
    if (!status.is_ok()) return status;
    *output = std::move(replacement);
    return core::Status::ok();
}

core::Status upload_norm(cuda::CudaExecutionContext* context,
                         const chapter01::Tensor& host,
                         cuda::CudaTensor* output) {
    if (host.dtype() != chapter01::DType::Float32 ||
        host.shape().size() != 1 || host.shape()[0] <= 0 ||
        host.raw_data() == nullptr) {
        return core::Status::error("DeepSeek2 CUDA output norm is invalid");
    }
    core::Status status = cuda::CudaTensor::create_contiguous(
        {static_cast<std::size_t>(host.shape()[0])},
        core::DType::Float32, output);
    if (status.is_ok()) status = context->upload(
        output, host.raw_data(), host.bytes());
    return status;
}

CudaDeepSeek2DenseLayerConfig attention_config(
    const chapter01::ModelMetadata& metadata) {
    const std::size_t nope =
        metadata.attention_key_length_mla - metadata.rope_dimension_count;
    float scale = 1.0F / std::sqrt(static_cast<float>(
        metadata.attention_key_length_mla));
    if (metadata.rope_scaling_type == "yarn" &&
        metadata.rope_scaling_factor > 1.0F &&
        metadata.rope_yarn_log_multiplier > 0.0F) {
        const float multiplier = metadata.rope_yarn_log_multiplier / 0.1F;
        const float mscale = 1.0F + 0.1F * multiplier *
            std::log(metadata.rope_scaling_factor);
        scale *= mscale * mscale;
    }
    return {metadata.hidden_size, metadata.attention_head_count, nope,
        metadata.rope_dimension_count, metadata.attention_value_length_mla,
        metadata.attention_kv_lora_rank, metadata.feed_forward_length,
        metadata.rms_norm_epsilon, metadata.rope_freq_base,
        {metadata.rope_scaling_factor,
         metadata.rope_scaling_original_context, 32.0F, 1.0F}, scale};
}

}  // namespace

struct RealDeepSeek2CudaModel::Impl final {
    chapter01::ModelMetadata metadata;
    CudaDeepSeek2DenseLayerConfig attention_config;
    cuda::CudaQuantizedMatrix embedding;
    cuda::CudaTensor output_norm;
    cuda::CudaQuantizedMatrix output_weight;
    CudaDeepSeek2DenseLayerWeights dense;
    std::vector<CudaDeepSeek2MoeWeights> moe;
    std::vector<CudaDeepSeek2MoeConfig> moe_configs;
};

struct RealDeepSeek2CudaContext::Impl final {
    std::shared_ptr<const RealDeepSeek2CudaModel::Impl> model;
    cuda::CudaExecutionContext execution;
    cuda::CudaTensor hidden;
    cuda::CudaTensor final_hidden;
    cuda::CudaTensor logits;
    CudaDeepSeek2DenseLayer dense;
    std::vector<CudaDeepSeek2MoeLayer> moe;
    std::size_t position = 0;
    std::size_t capacity_tokens = 0;

    void trim(std::size_t target) {
        dense.trim(target);
        for (auto& layer : moe) layer.trim(target);
    }

    core::Status forward_layers() {
        core::Status status = dense.forward(
            &execution, model->dense.view(), &hidden, position);
        for (std::size_t index = 0; status.is_ok() && index < moe.size(); ++index) {
            CudaDeepSeek2MoeRoute route;
            status = moe[index].forward(
                &execution, model->moe[index], &hidden, position, &route);
        }
        if (!status.is_ok()) trim(position);
        return status;
    }
};

struct RealDeepSeek2CudaBatchWorkspace::Impl final {
    std::shared_ptr<const RealDeepSeek2CudaModel::Impl> model;
    cuda::CudaExecutionContext execution;
    CudaDeepSeek2MoeBatchWorkspace moe;
    std::size_t batch_size = 0;
    std::size_t use_count = 0;
};

std::unique_ptr<RealDeepSeek2CudaModel> RealDeepSeek2CudaModel::load(
    const std::string& model_path,
    std::size_t layer_count,
    std::string* error) {
    chapter01::GgufModelIndex index;
    chapter01::DeepSeek2DecoderTensorLayout layout;
    auto data = std::make_shared<Impl>();
    if (!chapter01::GgufReader::read_index(model_path, &index, error) ||
        !chapter01::ModelMetadataAdapter::from_gguf(
            index.metadata, &data->metadata, error) ||
        !chapter01::ModelSchema::validate_deepseek2_gguf(
            data->metadata, index, &layout, error)) {
        return nullptr;
    }
    if (layer_count == 0) layer_count = data->metadata.layer_count;
    if (layer_count == 0 || layer_count > data->metadata.layer_count) {
        if (error != nullptr) *error = "DeepSeek2 CUDA layer count is invalid";
        return nullptr;
    }
    data->attention_config = attention_config(data->metadata);
    cuda::CudaExecutionContext upload_context;
    core::Status status = cuda::CudaExecutionContext::create(&upload_context);
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return nullptr;
    }
    chapter01::Tensor embedding;
    chapter01::Tensor output_norm;
    chapter01::Tensor output_weight;
    if (!load_named(model_path, index, layout.embedding, &embedding, error) ||
        !load_named(model_path, index, layout.output_norm, &output_norm, error) ||
        !load_named(model_path, index, layout.output, &output_weight, error)) {
        return nullptr;
    }
    status = upload_matrix(&upload_context, embedding, &data->embedding);
    if (status.is_ok()) status = upload_norm(
        &upload_context, output_norm, &data->output_norm);
    if (status.is_ok()) status = upload_matrix(
        &upload_context, output_weight, &data->output_weight);
    chapter01::RealDeepSeek2DenseLayerWeights dense;
    if (status.is_ok() && !chapter01::RealDeepSeek2DenseLayerLoader::load(
            model_path, index, layout, 0, &dense, error)) return nullptr;
    if (status.is_ok()) status = CudaDeepSeek2DenseLayerWeights::upload(
        &upload_context, dense, &data->dense);
    data->moe.reserve(layer_count > 0 ? layer_count - 1 : 0);
    data->moe_configs.reserve(layer_count > 0 ? layer_count - 1 : 0);
    for (std::size_t layer = 1; status.is_ok() && layer < layer_count; ++layer) {
        chapter01::RealDeepSeek2MoeLayerWeights host;
        if (!chapter01::RealDeepSeek2MoeLayerLoader::load(
                model_path, index, layout, layer, &host, error)) return nullptr;
        CudaDeepSeek2MoeWeights device;
        status = CudaDeepSeek2MoeWeights::upload(
            &upload_context, host, &device);
        if (status.is_ok()) {
            data->moe_configs.push_back({data->metadata.hidden_size,
                device.expert_hidden_size(),
                device.shared_expert_hidden_size(), data->metadata.expert_count,
                data->metadata.expert_used_count,
                data->metadata.expert_weights_normalize,
                data->metadata.expert_weights_scale});
            data->moe.push_back(std::move(device));
        }
    }
    if (status.is_ok()) status = upload_context.synchronize();
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return nullptr;
    }
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealDeepSeek2CudaModel>(
        new RealDeepSeek2CudaModel(std::move(data)));
}

RealDeepSeek2CudaModel::RealDeepSeek2CudaModel(
    std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
RealDeepSeek2CudaModel::~RealDeepSeek2CudaModel() = default;
RealDeepSeek2CudaModel::RealDeepSeek2CudaModel(
    RealDeepSeek2CudaModel&&) noexcept = default;
RealDeepSeek2CudaModel& RealDeepSeek2CudaModel::operator=(
    RealDeepSeek2CudaModel&&) noexcept = default;

std::unique_ptr<RealDeepSeek2CudaContext>
RealDeepSeek2CudaModel::create_context(
    std::size_t capacity_tokens, std::string* error) const {
    if (capacity_tokens == 0 || capacity_tokens > impl_->metadata.max_position) {
        if (error != nullptr) *error = "DeepSeek2 CUDA context capacity is invalid";
        return nullptr;
    }
    auto context = std::make_unique<RealDeepSeek2CudaContext::Impl>();
    context->model = impl_;
    context->capacity_tokens = capacity_tokens;
    core::Status status = cuda::CudaExecutionContext::create(&context->execution);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {impl_->metadata.hidden_size}, core::DType::Float32, &context->hidden);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {impl_->metadata.hidden_size}, core::DType::Float32,
        &context->final_hidden);
    if (status.is_ok()) status = cuda::CudaTensor::create_contiguous(
        {impl_->metadata.vocab_size}, core::DType::Float32, &context->logits);
    if (status.is_ok()) status = CudaDeepSeek2DenseLayer::create(
        impl_->attention_config, capacity_tokens, &context->dense);
    context->moe.reserve(impl_->moe.size());
    for (std::size_t index = 0;
         status.is_ok() && index < impl_->moe.size(); ++index) {
        CudaDeepSeek2MoeLayer layer;
        status = CudaDeepSeek2MoeLayer::create(
            {impl_->attention_config, impl_->moe_configs[index]},
            capacity_tokens, &layer);
        if (status.is_ok()) context->moe.push_back(std::move(layer));
    }
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return nullptr;
    }
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealDeepSeek2CudaContext>(
        new RealDeepSeek2CudaContext(std::move(context)));
}

std::unique_ptr<RealDeepSeek2CudaBatchWorkspace>
RealDeepSeek2CudaModel::create_batch_workspace(
    std::size_t batch_size, std::string* error) const {
    if (batch_size == 0 || impl_->moe_configs.empty()) {
        if (error != nullptr) {
            *error = "DeepSeek2 CUDA batch size or MoE layers are invalid";
        }
        return nullptr;
    }
    auto workspace = std::make_unique<RealDeepSeek2CudaBatchWorkspace::Impl>();
    workspace->model = impl_;
    workspace->batch_size = batch_size;
    core::Status status = cuda::CudaExecutionContext::create(
        &workspace->execution);
    if (status.is_ok()) status = CudaDeepSeek2MoeBatchWorkspace::create(
        impl_->moe_configs.front(), batch_size, &workspace->moe);
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return nullptr;
    }
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealDeepSeek2CudaBatchWorkspace>(
        new RealDeepSeek2CudaBatchWorkspace(std::move(workspace)));
}

bool RealDeepSeek2CudaModel::decode_batch(
    const std::vector<RealDeepSeek2CudaBatchItem>& items,
    RealDeepSeek2CudaBatchWorkspace* workspace,
    std::vector<std::vector<float>>* logits,
    std::string* error) const {
    if (items.empty() || workspace == nullptr || workspace->impl_ == nullptr ||
        logits == nullptr || workspace->impl_->model != impl_ ||
        workspace->impl_->batch_size != items.size()) {
        if (error != nullptr) {
            *error = "DeepSeek2 CUDA batch requires matching inputs and workspace";
        }
        return false;
    }
    std::unordered_set<RealDeepSeek2CudaContext*> unique;
    std::vector<std::size_t> positions;
    positions.reserve(items.size());
    for (const auto& item : items) {
        if (item.context == nullptr || !unique.insert(item.context).second ||
            item.context->impl_->model != impl_ ||
            item.context->impl_->position == 0 ||
            item.context->impl_->position >=
                item.context->impl_->capacity_tokens ||
            item.token < 0 ||
            static_cast<std::size_t>(item.token) >= impl_->metadata.vocab_size) {
            if (error != nullptr) {
                *error = "DeepSeek2 CUDA batch requires unique prefilled contexts";
            }
            return false;
        }
        const auto status = item.context->impl_->execution.synchronize();
        if (!status.is_ok()) {
            if (error != nullptr) *error = status.message();
            return false;
        }
        positions.push_back(item.context->impl_->position);
    }
    const auto rollback = [&]() {
        for (std::size_t index = 0; index < items.size(); ++index) {
            items[index].context->impl_->trim(positions[index]);
        }
    };

    core::Status status = core::Status::ok();
    for (std::size_t index = 0; status.is_ok() && index < items.size(); ++index) {
        auto& context = *items[index].context->impl_;
        status = context.execution.embedding_lookup(
            impl_->embedding, static_cast<std::size_t>(items[index].token),
            &context.hidden);
        if (status.is_ok()) status = context.dense.forward(
            &context.execution, impl_->dense.view(), &context.hidden,
            positions[index]);
    }
    for (std::size_t layer = 0;
         status.is_ok() && layer < impl_->moe.size(); ++layer) {
        for (std::size_t index = 0;
             status.is_ok() && index < items.size(); ++index) {
            auto& context = *items[index].context->impl_;
            status = context.moe[layer].forward_attention(
                &context.execution, impl_->moe[layer], &context.hidden,
                positions[index]);
            if (status.is_ok()) status = context.execution.synchronize();
        }
        std::vector<const cuda::CudaTensor*> normalized;
        std::vector<cuda::CudaTensor*> outputs;
        normalized.reserve(items.size());
        outputs.reserve(items.size());
        for (const auto& item : items) {
            normalized.push_back(
                &item.context->impl_->moe[layer].normalized_ffn_input());
            outputs.push_back(item.context->impl_->moe[layer].moe_output());
        }
        std::vector<CudaDeepSeek2MoeRoute> routes;
        if (status.is_ok()) status = workspace->impl_->moe.forward(
            &workspace->impl_->execution, impl_->moe[layer], normalized,
            outputs, &routes);
        for (std::size_t index = 0;
             status.is_ok() && index < items.size(); ++index) {
            auto& context = *items[index].context->impl_;
            status = context.moe[layer].finish_moe(
                &context.execution, &context.hidden);
        }
    }

    std::vector<std::vector<float>> built(
        items.size(), std::vector<float>(impl_->metadata.vocab_size));
    for (std::size_t index = 0; status.is_ok() && index < items.size(); ++index) {
        auto& context = *items[index].context->impl_;
        status = context.execution.rms_norm(
            context.hidden, impl_->output_norm, &context.final_hidden,
            impl_->metadata.rms_norm_epsilon);
        if (status.is_ok()) status = context.execution.quantized_linear(
            impl_->output_weight, context.final_hidden, &context.logits);
        if (status.is_ok()) status = context.execution.download(
            context.logits, built[index].data(),
            built[index].size() * sizeof(float));
        if (status.is_ok()) status = context.execution.synchronize();
    }
    if (!status.is_ok()) {
        rollback();
        if (error != nullptr) *error = status.message();
        return false;
    }
    for (const auto& item : items) ++item.context->impl_->position;
    *logits = std::move(built);
    ++workspace->impl_->use_count;
    if (error != nullptr) error->clear();
    return true;
}

std::size_t RealDeepSeek2CudaModel::layer_count() const {
    return 1 + impl_->moe.size();
}
std::size_t RealDeepSeek2CudaModel::hidden_size() const {
    return impl_->metadata.hidden_size;
}
std::size_t RealDeepSeek2CudaModel::vocab_size() const {
    return impl_->metadata.vocab_size;
}
std::size_t RealDeepSeek2CudaModel::max_context_tokens() const {
    return impl_->metadata.max_position;
}
std::int32_t RealDeepSeek2CudaModel::eos_token_id() const {
    return impl_->metadata.eos_token_id;
}

RealDeepSeek2CudaBatchWorkspace::RealDeepSeek2CudaBatchWorkspace(
    std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RealDeepSeek2CudaBatchWorkspace::~RealDeepSeek2CudaBatchWorkspace() = default;
RealDeepSeek2CudaBatchWorkspace::RealDeepSeek2CudaBatchWorkspace(
    RealDeepSeek2CudaBatchWorkspace&&) noexcept = default;
RealDeepSeek2CudaBatchWorkspace&
RealDeepSeek2CudaBatchWorkspace::operator=(
    RealDeepSeek2CudaBatchWorkspace&&) noexcept = default;
std::size_t RealDeepSeek2CudaBatchWorkspace::batch_size() const {
    return impl_ == nullptr ? 0 : impl_->batch_size;
}
std::size_t RealDeepSeek2CudaBatchWorkspace::use_count() const {
    return impl_ == nullptr ? 0 : impl_->use_count;
}
std::size_t RealDeepSeek2CudaBatchWorkspace::grouped_moe_calls() const {
    return impl_ == nullptr ? 0 : impl_->moe.use_count();
}
std::size_t RealDeepSeek2CudaBatchWorkspace::active_expert_groups() const {
    return impl_ == nullptr ? 0 : impl_->moe.active_expert_groups();
}
std::size_t RealDeepSeek2CudaBatchWorkspace::multi_row_expert_groups() const {
    return impl_ == nullptr ? 0 : impl_->moe.multi_row_expert_groups();
}
std::size_t RealDeepSeek2CudaBatchWorkspace::routed_rows() const {
    return impl_ == nullptr ? 0 : impl_->moe.routed_rows();
}

RealDeepSeek2CudaContext::RealDeepSeek2CudaContext(
    std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RealDeepSeek2CudaContext::~RealDeepSeek2CudaContext() = default;
RealDeepSeek2CudaContext::RealDeepSeek2CudaContext(
    RealDeepSeek2CudaContext&&) noexcept = default;
RealDeepSeek2CudaContext& RealDeepSeek2CudaContext::operator=(
    RealDeepSeek2CudaContext&&) noexcept = default;

core::Status RealDeepSeek2CudaContext::forward_hidden(
    const std::vector<float>& input, std::vector<float>* output) {
    if (output == nullptr || input.size() != impl_->model->metadata.hidden_size ||
        impl_->position >= impl_->capacity_tokens) {
        return core::Status::error("DeepSeek2 CUDA hidden input is invalid");
    }
    core::Status status = impl_->execution.upload(
        &impl_->hidden, input.data(), input.size() * sizeof(float));
    if (status.is_ok()) status = impl_->forward_layers();
    std::vector<float> result(input.size());
    if (status.is_ok()) status = impl_->execution.download(
        impl_->hidden, result.data(), result.size() * sizeof(float));
    if (status.is_ok()) status = impl_->execution.synchronize();
    if (!status.is_ok()) {
        impl_->trim(impl_->position);
        return status;
    }
    ++impl_->position;
    *output = std::move(result);
    return core::Status::ok();
}

core::Status RealDeepSeek2CudaContext::forward_token(
    std::int32_t token, std::vector<float>* logits) {
    if (logits == nullptr || token < 0 ||
        static_cast<std::size_t>(token) >= impl_->model->metadata.vocab_size ||
        impl_->position >= impl_->capacity_tokens) {
        return core::Status::error("DeepSeek2 CUDA token input is invalid");
    }
    core::Status status = impl_->execution.embedding_lookup(
        impl_->model->embedding, static_cast<std::size_t>(token),
        &impl_->hidden);
    if (status.is_ok()) status = impl_->forward_layers();
    if (status.is_ok()) status = impl_->execution.rms_norm(
        impl_->hidden, impl_->model->output_norm, &impl_->final_hidden,
        impl_->model->metadata.rms_norm_epsilon);
    if (status.is_ok()) status = impl_->execution.quantized_linear(
        impl_->model->output_weight, impl_->final_hidden, &impl_->logits);
    std::vector<float> result(impl_->model->metadata.vocab_size);
    if (status.is_ok()) status = impl_->execution.download(
        impl_->logits, result.data(), result.size() * sizeof(float));
    if (status.is_ok()) status = impl_->execution.synchronize();
    if (!status.is_ok()) {
        impl_->trim(impl_->position);
        return status;
    }
    ++impl_->position;
    *logits = std::move(result);
    return core::Status::ok();
}

void RealDeepSeek2CudaContext::reset() {
    impl_->dense.reset();
    for (auto& layer : impl_->moe) layer.reset();
    impl_->position = 0;
}
std::size_t RealDeepSeek2CudaContext::position() const {
    return impl_->position;
}
std::size_t RealDeepSeek2CudaContext::layer_count() const {
    return 1 + impl_->moe.size();
}

}  // namespace cpp_llm_engine::model
