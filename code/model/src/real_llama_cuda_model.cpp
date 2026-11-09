// Llama CUDA Model：实现 GGUF 分层上传与请求级多层前向。
// 职责：组装 Chapter01 Loader、CUDA Decoder 以及连续或分页 K/V，不实现算子 kernel。
// 验证：真实 TinyLlama 首层 CPU/CUDA golden 与独立 Context 状态测试。
#include "cpp_llm_engine/model/real_llama_cuda_model.h"

#include "cpp_llm_engine/cuda/cuda_decoder_layer.h"
#include "cpp_llm_engine/cuda/cuda_decoder_batch_layer.h"
#include "cpp_llm_engine/cuda/cuda_execution_context.h"
#include "cpp_llm_engine/cuda/cuda_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_paged_kv_cache.h"
#include "cpp_llm_engine/cuda/cuda_tensor.h"
#include "cpp_llm_engine/model/cuda_llama_layer_weights.h"
#include "gguf_reader.h"
#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "model_schema.h"
#include "real_layer_weights.h"

#include <limits>
#include <memory>
#include <unordered_set>
#include <utility>

namespace cpp_llm_engine::model {
namespace {

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index,
    const std::string& name) {
    for (const auto& tensor : index.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool load_named(
    const std::string& model_path,
    const chapter01::GgufModelIndex& index,
    const std::string& name,
    chapter01::Tensor* output,
    std::string* error) {
    const auto* info = find_tensor(index, name);
    if (info == nullptr) {
        if (error != nullptr) *error = "missing CUDA model tensor: " + name;
        return false;
    }
    return chapter01::GgufTensorLoader::load_tensor(
        model_path, index, *info, output, error);
}

core::Status upload_norm(
    cuda::CudaExecutionContext* context,
    const chapter01::Tensor& host,
    cuda::CudaTensor* output) {
    if (context == nullptr || output == nullptr ||
        host.dtype() != chapter01::DType::Float32 ||
        host.shape().size() != 1 || host.shape()[0] <= 0 ||
        host.raw_data() == nullptr) {
        return core::Status::error("CUDA model norm tensor is invalid");
    }
    core::Status status = cuda::CudaTensor::create_contiguous(
        {static_cast<std::size_t>(host.shape()[0])},
        core::DType::Float32,
        output);
    if (!status.is_ok()) return status;
    return context->upload(output, host.raw_data(), host.bytes());
}

core::Status upload_quantized_matrix(
    cuda::CudaExecutionContext* context,
    const chapter01::Tensor& host,
    cuda::CudaQuantizedMatrix* output) {
    if (context == nullptr || output == nullptr || host.shape().size() != 2 ||
        host.shape()[0] <= 0 || host.shape()[1] <= 0 ||
        host.raw_data() == nullptr) {
        return core::Status::error("CUDA model quantized matrix is invalid");
    }
    core::WeightFormat format = core::WeightFormat::Unknown;
    if (host.dtype() == chapter01::DType::Q4_K) format = core::WeightFormat::Q4K;
    if (host.dtype() == chapter01::DType::Q6_K) format = core::WeightFormat::Q6K;
    if (format == core::WeightFormat::Unknown) {
        return core::Status::error("CUDA model matrix requires Q4_K or Q6_K");
    }
    const auto cols = static_cast<std::size_t>(host.shape()[0]);
    const auto rows = static_cast<std::size_t>(host.shape()[1]);
    core::Status status = cuda::CudaQuantizedMatrix::create(
        rows, cols, format, output);
    if (!status.is_ok()) return status;
    return context->upload(output, host.raw_data(), host.bytes());
}

}  // namespace

struct RealLlamaCudaModel::Impl final {
    chapter01::ModelMetadata metadata;
    chapter01::LlamaTensorLayout layout;
    cuda::CudaDecoderLayerConfig layer_config;
    cuda::CudaQuantizedMatrix embedding;
    cuda::CudaTensor output_norm;
    cuda::CudaQuantizedMatrix output_weight;
    std::vector<CudaLlamaLayerWeights> layers;
};

struct RealLlamaCudaContext::Impl final {
    std::shared_ptr<const RealLlamaCudaModel::Impl> model;
    cuda::CudaExecutionContext execution;
    cuda::CudaTensor hidden;
    cuda::CudaTensor final_hidden;
    cuda::CudaTensor logits;
    std::vector<cuda::CudaDecoderLayer> layers;
    std::vector<cuda::CudaKvCache> caches;
    std::shared_ptr<RealLlamaCudaKvDataPlane> external_kv;
    std::size_t position = 0;
    std::size_t capacity_tokens = 0;
    std::size_t prefill_token_count = 0;
    std::size_t prefill_workspace_creations = 0;
    cuda::CudaTensor prefill_hidden;
    std::vector<cuda::CudaTensor> prefill_rows;
    cuda::CudaDecoderBatchLayer prefill_layer;

    void rollback_caches() {
        if (external_kv != nullptr) external_kv->trim(position);
        for (auto& cache : caches) cache.trim(position);
    }

    core::Status prepare_prefill_workspace(std::size_t token_count) {
        if (prefill_token_count == token_count) return core::Status::ok();
        cuda::CudaTensor hidden_replacement;
        cuda::CudaDecoderBatchLayer layer_replacement;
        std::vector<cuda::CudaTensor> rows_replacement(token_count);
        core::Status status = cuda::CudaTensor::create_contiguous(
            {token_count, model->metadata.hidden_size},
            core::DType::Float32, &hidden_replacement);
        if (status.is_ok()) {
            status = cuda::CudaDecoderBatchLayer::create(
                model->layer_config, token_count, &layer_replacement);
        }
        for (std::size_t index = 0; status.is_ok() && index < token_count; ++index) {
            status = cuda::CudaTensor::create_contiguous(
                {model->metadata.hidden_size}, core::DType::Float32,
                &rows_replacement[index]);
        }
        if (!status.is_ok()) return status;
        prefill_hidden = std::move(hidden_replacement);
        prefill_layer = std::move(layer_replacement);
        prefill_rows = std::move(rows_replacement);
        prefill_token_count = token_count;
        ++prefill_workspace_creations;
        return core::Status::ok();
    }

    core::Status forward_layers() {
        if (external_kv != nullptr) {
            std::string error;
            if (!external_kv->reserve(position + 1, &error)) {
                return core::Status::error(error);
            }
            const std::vector<std::size_t> blocks = external_kv->physical_blocks();
            for (std::size_t index = 0; index < layers.size(); ++index) {
                const core::Status status = layers[index].forward_paged(
                    &execution,
                    model->layers[index].decoder_weights(),
                    external_kv->payload(),
                    index,
                    blocks,
                    &hidden,
                    position);
                if (!status.is_ok()) {
                    external_kv->trim(position);
                    return status;
                }
            }
            return core::Status::ok();
        }
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const core::Status status = layers[index].forward(
                &execution,
                model->layers[index].decoder_weights(),
                &caches[index],
                &hidden,
                position);
            if (!status.is_ok()) {
                rollback_caches();
                return status;
            }
        }
        return core::Status::ok();
    }
};

struct RealLlamaCudaBatchWorkspace::Impl final {
    std::shared_ptr<const RealLlamaCudaModel::Impl> model;
    cuda::CudaExecutionContext execution;
    cuda::CudaTensor hidden;
    cuda::CudaTensor final_hidden;
    cuda::CudaTensor logits;
    std::vector<cuda::CudaTensor> embedding_rows;
    cuda::CudaDecoderBatchLayer decoder_layer;
    std::size_t batch_size = 0;
    std::size_t use_count = 0;
};

void RealLlamaCudaContext::set_attention_path(cuda::CudaAttentionPath path) {
    impl_->execution.set_attention_path(path);
}

std::unique_ptr<RealLlamaCudaModel> RealLlamaCudaModel::load(
    const std::string& model_path,
    std::size_t layer_count,
    std::string* error) {
    chapter01::GgufModelIndex index;
    auto data = std::make_shared<Impl>();
    if (!chapter01::GgufReader::read_index(model_path, &index, error) ||
        !chapter01::ModelMetadataAdapter::from_gguf(
            index.metadata, &data->metadata, error) ||
        !chapter01::ModelSchema::validate_gguf(
            data->metadata, index, &data->layout, error)) {
        return nullptr;
    }
    if (layer_count == 0) layer_count = data->metadata.layer_count;
    if (layer_count == 0 || layer_count > data->metadata.layer_count) {
        if (error != nullptr) *error = "requested CUDA decoder layer count is invalid";
        return nullptr;
    }
    const std::size_t query_heads = data->metadata.attention_head_count;
    const std::size_t kv_heads = data->metadata.attention_head_count_kv;
    if (query_heads == 0 || kv_heads == 0 || query_heads % kv_heads != 0 ||
        data->metadata.hidden_size % query_heads != 0) {
        if (error != nullptr) *error = "CUDA attention head metadata is invalid";
        return nullptr;
    }
    const std::size_t head_dimension = data->metadata.hidden_size / query_heads;
    const std::size_t rotary_dimension = data->metadata.rope_dimension_count == 0
        ? head_dimension : data->metadata.rope_dimension_count;
    data->layer_config = {
        data->metadata.hidden_size,
        data->layout.intermediate_size,
        query_heads,
        kv_heads,
        head_dimension,
        rotary_dimension,
        data->metadata.rms_norm_epsilon,
        data->metadata.rope_freq_base,
    };

    cuda::CudaExecutionContext upload_context;
    core::Status status = cuda::CudaExecutionContext::create(&upload_context);
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return nullptr;
    }
    chapter01::Tensor embedding;
    chapter01::Tensor output_norm;
    chapter01::Tensor output_weight;
    if (!load_named(
            model_path, index, data->layout.embedding, &embedding, error) ||
        !load_named(
            model_path, index, data->layout.output_norm, &output_norm, error) ||
        !load_named(
            model_path, index, data->layout.output, &output_weight, error)) {
        return nullptr;
    }
    status = upload_quantized_matrix(
        &upload_context, embedding, &data->embedding);
    if (status.is_ok()) {
        status = upload_norm(&upload_context, output_norm, &data->output_norm);
    }
    if (status.is_ok()) {
        status = upload_quantized_matrix(
            &upload_context, output_weight, &data->output_weight);
    }
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return nullptr;
    }
    data->layers.reserve(layer_count);
    for (std::size_t index_value = 0; index_value < layer_count; ++index_value) {
        chapter01::RealLlamaLayerWeights host;
        if (!chapter01::RealLlamaLayerLoader::load(
                model_path, index, data->layout, index_value, &host, error)) {
            return nullptr;
        }
        CudaLlamaLayerWeights device;
        status = CudaLlamaLayerWeights::upload(&upload_context, host, &device);
        if (!status.is_ok()) {
            if (error != nullptr) *error = status.message();
            return nullptr;
        }
        data->layers.push_back(std::move(device));
    }
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaCudaModel>(
        new RealLlamaCudaModel(std::move(data)));
}

RealLlamaCudaModel::RealLlamaCudaModel(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCudaModel::~RealLlamaCudaModel() = default;
RealLlamaCudaModel::RealLlamaCudaModel(RealLlamaCudaModel&&) noexcept = default;
RealLlamaCudaModel& RealLlamaCudaModel::operator=(RealLlamaCudaModel&&) noexcept = default;

std::unique_ptr<RealLlamaCudaContext> RealLlamaCudaModel::create_context(
    std::size_t capacity_tokens,
    std::string* error) const {
    if (capacity_tokens == 0 ||
        capacity_tokens > impl_->metadata.max_position) {
        if (error != nullptr) *error = "CUDA context capacity is invalid";
        return nullptr;
    }
    auto context = std::make_unique<RealLlamaCudaContext::Impl>();
    context->model = impl_;
    context->capacity_tokens = capacity_tokens;
    core::Status status = cuda::CudaExecutionContext::create(&context->execution);
    if (status.is_ok()) {
        status = cuda::CudaTensor::create_contiguous(
            {impl_->metadata.hidden_size}, core::DType::Float32, &context->hidden);
    }
    if (status.is_ok()) {
        status = cuda::CudaTensor::create_contiguous(
            {impl_->metadata.hidden_size},
            core::DType::Float32,
            &context->final_hidden);
    }
    if (status.is_ok()) {
        status = cuda::CudaTensor::create_contiguous(
            {impl_->metadata.vocab_size}, core::DType::Float32, &context->logits);
    }
    context->layers.reserve(impl_->layers.size());
    context->caches.reserve(impl_->layers.size());
    for (std::size_t index = 0; status.is_ok() && index < impl_->layers.size(); ++index) {
        cuda::CudaDecoderLayer layer;
        cuda::CudaKvCache cache;
        status = cuda::CudaDecoderLayer::create(impl_->layer_config, &layer);
        if (status.is_ok()) {
            status = cuda::CudaKvCache::create(
                capacity_tokens,
                impl_->layer_config.kv_heads,
                impl_->layer_config.head_dimension,
                &cache);
        }
        if (status.is_ok()) {
            context->layers.push_back(std::move(layer));
            context->caches.push_back(std::move(cache));
        }
    }
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return nullptr;
    }
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaCudaContext>(
        new RealLlamaCudaContext(std::move(context)));
}

std::unique_ptr<RealLlamaCudaBatchWorkspace>
RealLlamaCudaModel::create_batch_workspace(
    std::size_t batch_size,
    std::string* error) const {
    if (batch_size == 0) {
        if (error != nullptr) *error = "CUDA batch workspace size is invalid";
        return nullptr;
    }
    auto workspace = std::make_unique<RealLlamaCudaBatchWorkspace::Impl>();
    workspace->model = impl_;
    workspace->batch_size = batch_size;
    core::Status status = cuda::CudaExecutionContext::create(&workspace->execution);
    if (status.is_ok()) {
        status = cuda::CudaTensor::create_contiguous(
            {batch_size, hidden_size()}, core::DType::Float32, &workspace->hidden);
    }
    if (status.is_ok()) {
        status = cuda::CudaTensor::create_contiguous(
            {batch_size, hidden_size()},
            core::DType::Float32,
            &workspace->final_hidden);
    }
    if (status.is_ok()) {
        status = cuda::CudaTensor::create_contiguous(
            {batch_size, vocab_size()}, core::DType::Float32, &workspace->logits);
    }
    workspace->embedding_rows.resize(batch_size);
    for (std::size_t index = 0; status.is_ok() && index < batch_size; ++index) {
        status = cuda::CudaTensor::create_contiguous(
            {hidden_size()}, core::DType::Float32, &workspace->embedding_rows[index]);
    }
    if (status.is_ok()) {
        status = cuda::CudaDecoderBatchLayer::create(
            impl_->layer_config, batch_size, &workspace->decoder_layer);
    }
    if (status.is_ok()) {
        const std::size_t max_tokens = impl_->metadata.max_position;
        const std::size_t query_heads = impl_->layer_config.query_heads;
        const std::size_t maximum = std::numeric_limits<std::size_t>::max();
        if (query_heads > maximum / batch_size ||
            max_tokens > maximum / (batch_size * query_heads) ||
            batch_size * query_heads * max_tokens > maximum / sizeof(float) ||
            max_tokens > maximum / sizeof(std::size_t) ||
            batch_size * query_heads * max_tokens * sizeof(float) >
                maximum - max_tokens * sizeof(std::size_t)) {
            status = core::Status::error(
                "CUDA batch workspace attention capacity overflows");
        } else {
            const std::size_t score_bytes =
                batch_size * query_heads * max_tokens * sizeof(float);
            const std::size_t page_bytes = max_tokens * sizeof(std::size_t);
            status = workspace->execution.reserve_workspace(
                score_bytes + page_bytes);
        }
    }
    if (!status.is_ok()) {
        if (error != nullptr) *error = status.message();
        return nullptr;
    }
    if (error != nullptr) error->clear();
    return std::unique_ptr<RealLlamaCudaBatchWorkspace>(
        new RealLlamaCudaBatchWorkspace(std::move(workspace)));
}

bool RealLlamaCudaModel::decode_batch(
    const std::vector<RealLlamaCudaBatchItem>& items,
    RealLlamaCudaBatchWorkspace* workspace,
    std::vector<std::vector<float>>* logits,
    std::string* error) const {
    if (items.empty() || workspace == nullptr || workspace->impl_ == nullptr ||
        logits == nullptr || workspace->impl_->model != impl_ ||
        workspace->impl_->batch_size != items.size()) {
        if (error != nullptr) {
            *error = "CUDA decode batch requires matching items, workspace, and output";
        }
        return false;
    }
    auto& storage = *workspace->impl_;
    std::unordered_set<RealLlamaCudaContext*> unique;
    std::vector<std::size_t> positions;
    positions.reserve(items.size());
    const bool uses_external_kv = items.front().context != nullptr &&
        items.front().context->impl_->external_kv != nullptr;
    cuda::CudaPagedKvCache* paged_payload = nullptr;
    std::vector<std::vector<std::size_t>> physical_blocks;
    if (uses_external_kv) physical_blocks.reserve(items.size());
    for (const auto& item : items) {
        if (item.context == nullptr || !unique.insert(item.context).second ||
            item.context->impl_->model != impl_ || item.context->impl_->position == 0 ||
            item.context->impl_->position >= item.context->impl_->capacity_tokens ||
            item.token < 0 || static_cast<std::size_t>(item.token) >= vocab_size()) {
            if (error != nullptr) {
                *error = "CUDA decode batch requires unique prefilled contexts and valid tokens";
            }
            return false;
        }
        const bool item_uses_external_kv = item.context->impl_->external_kv != nullptr;
        if (item_uses_external_kv != uses_external_kv) {
            if (error != nullptr) {
                *error = "CUDA decode batch cannot mix continuous and paged contexts";
            }
            return false;
        }
        if (uses_external_kv) {
            auto* payload = item.context->impl_->external_kv->payload();
            const auto blocks = item.context->impl_->external_kv->physical_blocks();
            if (payload == nullptr ||
                (paged_payload != nullptr && payload != paged_payload) ||
                item.context->impl_->external_kv->token_count() !=
                    item.context->impl_->position ||
                blocks.empty() || item.context->impl_->position /
                    payload->block_size() >= blocks.size()) {
                if (error != nullptr) {
                    *error = "CUDA paged batch data plane is not aligned or reserved";
                }
                return false;
            }
            paged_payload = payload;
            physical_blocks.push_back(blocks);
        }
        const core::Status sync_status = item.context->impl_->execution.synchronize();
        if (!sync_status.is_ok()) {
            if (error != nullptr) *error = sync_status.message();
            return false;
        }
        positions.push_back(item.context->impl_->position);
    }

    core::Status status = core::Status::ok();
    std::vector<const cuda::CudaTensor*> embedding_row_ptrs;
    embedding_row_ptrs.reserve(items.size());
    for (std::size_t index = 0; status.is_ok() && index < items.size(); ++index) {
        status = storage.execution.embedding_lookup(
            impl_->embedding,
            static_cast<std::size_t>(items[index].token),
            &storage.embedding_rows[index]);
        embedding_row_ptrs.push_back(&storage.embedding_rows[index]);
    }
    if (status.is_ok()) {
        status = storage.execution.pack_rows(embedding_row_ptrs, &storage.hidden);
    }
    const auto rollback = [&]() {
        for (std::size_t item_index = 0; item_index < items.size(); ++item_index) {
            if (uses_external_kv) {
                (void)items[item_index].context->impl_->external_kv->trim(
                    positions[item_index]);
            } else {
                for (auto& cache : items[item_index].context->impl_->caches) {
                    (void)cache.trim(positions[item_index]);
                }
            }
        }
    };
    for (std::size_t layer = 0; status.is_ok() && layer < impl_->layers.size(); ++layer) {
        if (uses_external_kv) {
            status = storage.decoder_layer.forward_paged(
                &storage.execution, impl_->layers[layer].decoder_weights(),
                paged_payload, layer, physical_blocks, &storage.hidden, positions);
        } else {
            std::vector<cuda::CudaKvCache*> caches;
            caches.reserve(items.size());
            for (const auto& item : items) {
                caches.push_back(&item.context->impl_->caches[layer]);
            }
            status = storage.decoder_layer.forward(
                &storage.execution, impl_->layers[layer].decoder_weights(),
                caches, &storage.hidden, positions);
        }
    }
    if (status.is_ok()) {
        status = storage.execution.rms_norm_batch(
            storage.hidden, impl_->output_norm, &storage.final_hidden,
            impl_->metadata.rms_norm_epsilon);
    }
    if (status.is_ok()) {
        status = storage.execution.quantized_linear_batch(
            impl_->output_weight, storage.final_hidden, &storage.logits);
    }
    std::vector<float> flat_logits(items.size() * vocab_size());
    if (status.is_ok()) {
        status = storage.execution.download(
            storage.logits, flat_logits.data(), flat_logits.size() * sizeof(float));
    }
    if (status.is_ok()) status = storage.execution.synchronize();
    if (!status.is_ok()) {
        rollback();
        if (error != nullptr) *error = status.message();
        return false;
    }

    std::vector<std::vector<float>> result(items.size());
    for (std::size_t index = 0; index < items.size(); ++index) {
        result[index].assign(
            flat_logits.begin() + static_cast<std::ptrdiff_t>(index * vocab_size()),
            flat_logits.begin() + static_cast<std::ptrdiff_t>((index + 1) * vocab_size()));
        ++items[index].context->impl_->position;
    }
    *logits = std::move(result);
    ++storage.use_count;
    if (error != nullptr) error->clear();
    return true;
}

RealLlamaCudaBatchWorkspace::RealLlamaCudaBatchWorkspace(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCudaBatchWorkspace::~RealLlamaCudaBatchWorkspace() = default;
RealLlamaCudaBatchWorkspace::RealLlamaCudaBatchWorkspace(
    RealLlamaCudaBatchWorkspace&&) noexcept = default;
RealLlamaCudaBatchWorkspace& RealLlamaCudaBatchWorkspace::operator=(
    RealLlamaCudaBatchWorkspace&&) noexcept = default;

std::size_t RealLlamaCudaBatchWorkspace::batch_size() const {
    return impl_->batch_size;
}

std::size_t RealLlamaCudaBatchWorkspace::use_count() const {
    return impl_->use_count;
}

std::size_t RealLlamaCudaBatchWorkspace::device_bytes() const {
    std::size_t bytes = impl_->hidden.size_bytes() + impl_->final_hidden.size_bytes() +
        impl_->logits.size_bytes() + impl_->decoder_layer.device_bytes() +
        impl_->execution.workspace_capacity_bytes();
    for (const auto& row : impl_->embedding_rows) bytes += row.size_bytes();
    return bytes;
}

std::size_t RealLlamaCudaBatchWorkspace::dynamic_allocation_count() const {
    return impl_->execution.workspace_allocation_count();
}

std::size_t RealLlamaCudaModel::layer_count() const { return impl_->layers.size(); }
std::size_t RealLlamaCudaModel::model_layer_count() const {
    return impl_->metadata.layer_count;
}
std::size_t RealLlamaCudaModel::hidden_size() const {
    return impl_->metadata.hidden_size;
}
std::size_t RealLlamaCudaModel::kv_head_count() const {
    return impl_->layer_config.kv_heads;
}
std::size_t RealLlamaCudaModel::head_dimension() const {
    return impl_->layer_config.head_dimension;
}
std::size_t RealLlamaCudaModel::vocab_size() const {
    return impl_->metadata.vocab_size;
}
std::size_t RealLlamaCudaModel::max_context_tokens() const {
    return impl_->metadata.max_position;
}
std::int32_t RealLlamaCudaModel::eos_token_id() const {
    return impl_->metadata.eos_token_id;
}

RealLlamaCudaContext::RealLlamaCudaContext(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RealLlamaCudaContext::~RealLlamaCudaContext() = default;
RealLlamaCudaContext::RealLlamaCudaContext(RealLlamaCudaContext&&) noexcept = default;
RealLlamaCudaContext& RealLlamaCudaContext::operator=(RealLlamaCudaContext&&) noexcept = default;

core::Status RealLlamaCudaContext::forward_hidden(
    const std::vector<float>& input,
    std::vector<float>* output) {
    if (output == nullptr || input.size() != impl_->model->metadata.hidden_size) {
        return core::Status::error("CUDA hidden input or output is invalid");
    }
    if (impl_->position >= impl_->capacity_tokens) {
        return core::Status::error("CUDA context KV capacity is exhausted");
    }
    core::Status status = impl_->execution.upload(
        &impl_->hidden, input.data(), input.size() * sizeof(float));
    if (!status.is_ok()) return status;
    status = impl_->forward_layers();
    if (!status.is_ok()) return status;
    std::vector<float> result(input.size());
    status = impl_->execution.download(
        impl_->hidden, result.data(), result.size() * sizeof(float));
    if (status.is_ok()) status = impl_->execution.synchronize();
    if (!status.is_ok()) {
        impl_->rollback_caches();
        return status;
    }
    if (impl_->external_kv != nullptr) {
        std::string error;
        if (!impl_->external_kv->commit(impl_->position + 1, &error)) {
            impl_->rollback_caches();
            return core::Status::error(error);
        }
    }
    ++impl_->position;
    *output = std::move(result);
    return core::Status::ok();
}

core::Status RealLlamaCudaContext::forward_hidden_sequence(
    const std::vector<float>& input,
    std::size_t token_count,
    std::vector<float>* output) {
    const std::size_t hidden_size = impl_->model->metadata.hidden_size;
    if (output == nullptr || token_count == 0 ||
        token_count > impl_->capacity_tokens - impl_->position ||
        hidden_size > std::numeric_limits<std::size_t>::max() / token_count ||
        input.size() != token_count * hidden_size) {
        return core::Status::error("CUDA hidden sequence input or output is invalid");
    }
    std::vector<std::size_t> physical_blocks;
    if (impl_->external_kv != nullptr) {
        std::string error;
        if (!impl_->external_kv->reserve(impl_->position + token_count, &error)) {
            return core::Status::error(error);
        }
        physical_blocks = impl_->external_kv->physical_blocks();
    }
    cuda::CudaTensor hidden;
    cuda::CudaDecoderBatchLayer layer;
    core::Status status = cuda::CudaTensor::create_contiguous(
        {token_count, hidden_size}, core::DType::Float32, &hidden);
    if (status.is_ok()) {
        status = cuda::CudaDecoderBatchLayer::create(
            impl_->model->layer_config, token_count, &layer);
    }
    if (status.is_ok()) {
        status = impl_->execution.upload(
            &hidden, input.data(), input.size() * sizeof(float));
    }
    for (std::size_t index = 0; status.is_ok() && index < impl_->layers.size(); ++index) {
        if (impl_->external_kv != nullptr) {
            status = layer.prefill_sequence_paged(
                &impl_->execution,
                impl_->model->layers[index].decoder_weights(),
                impl_->external_kv->payload(), index, physical_blocks,
                &hidden, impl_->position);
        } else {
            status = layer.prefill_sequence(
                &impl_->execution,
                impl_->model->layers[index].decoder_weights(),
                &impl_->caches[index], &hidden, impl_->position);
        }
    }
    std::vector<float> result(input.size());
    if (status.is_ok()) {
        status = impl_->execution.download(
            hidden, result.data(), result.size() * sizeof(float));
    }
    if (status.is_ok()) status = impl_->execution.synchronize();
    if (!status.is_ok()) {
        impl_->rollback_caches();
        return status;
    }
    if (impl_->external_kv != nullptr) {
        std::string error;
        if (!impl_->external_kv->commit(impl_->position + token_count, &error)) {
            impl_->rollback_caches();
            return core::Status::error(error);
        }
    }
    impl_->position += token_count;
    *output = std::move(result);
    return core::Status::ok();
}

core::Status RealLlamaCudaContext::forward_token(
    std::int32_t token,
    std::vector<float>* logits) {
    if (logits == nullptr || token < 0 ||
        static_cast<std::size_t>(token) >= impl_->model->metadata.vocab_size) {
        return core::Status::error("CUDA token or logits output is invalid");
    }
    if (impl_->position >= impl_->capacity_tokens) {
        return core::Status::error("CUDA context KV capacity is exhausted");
    }
    core::Status status = impl_->execution.embedding_lookup(
        impl_->model->embedding,
        static_cast<std::size_t>(token),
        &impl_->hidden);
    if (status.is_ok()) status = impl_->forward_layers();
    if (status.is_ok()) {
        status = impl_->execution.rms_norm(
            impl_->hidden,
            impl_->model->output_norm,
            &impl_->final_hidden,
            impl_->model->metadata.rms_norm_epsilon);
    }
    if (status.is_ok()) {
        status = impl_->execution.quantized_linear(
            impl_->model->output_weight,
            impl_->final_hidden,
            &impl_->logits);
    }
    std::vector<float> result(impl_->model->metadata.vocab_size);
    if (status.is_ok()) {
        status = impl_->execution.download(
            impl_->logits,
            result.data(),
            result.size() * sizeof(float));
    }
    if (status.is_ok()) status = impl_->execution.synchronize();
    if (!status.is_ok()) {
        impl_->rollback_caches();
        return status;
    }
    if (impl_->external_kv != nullptr) {
        std::string error;
        if (!impl_->external_kv->commit(impl_->position + 1, &error)) {
            impl_->rollback_caches();
            return core::Status::error(error);
        }
    }
    ++impl_->position;
    *logits = std::move(result);
    return core::Status::ok();
}

core::Status RealLlamaCudaContext::prefill_tokens(
    const std::vector<std::int32_t>& tokens,
    std::vector<float>* logits) {
    if (logits == nullptr || tokens.empty() ||
        tokens.size() > impl_->capacity_tokens - impl_->position) {
        return core::Status::error("CUDA prefill tokens or output is invalid");
    }
    for (const std::int32_t token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >=
                             impl_->model->metadata.vocab_size) {
            return core::Status::error("CUDA prefill token is out of range");
        }
    }
    core::Status status = impl_->prepare_prefill_workspace(tokens.size());
    std::vector<const cuda::CudaTensor*> row_inputs;
    std::vector<cuda::CudaTensor*> row_outputs;
    row_inputs.reserve(tokens.size());
    row_outputs.reserve(tokens.size());
    for (std::size_t index = 0; status.is_ok() && index < tokens.size(); ++index) {
        status = impl_->execution.embedding_lookup(
            impl_->model->embedding, static_cast<std::size_t>(tokens[index]),
            &impl_->prefill_rows[index]);
        row_inputs.push_back(&impl_->prefill_rows[index]);
        row_outputs.push_back(&impl_->prefill_rows[index]);
    }
    if (status.is_ok()) {
        status = impl_->execution.pack_rows(row_inputs, &impl_->prefill_hidden);
    }
    std::vector<std::size_t> physical_blocks;
    if (status.is_ok() && impl_->external_kv != nullptr) {
        std::string error;
        if (!impl_->external_kv->reserve(impl_->position + tokens.size(), &error)) {
            status = core::Status::error(error);
        } else {
            physical_blocks = impl_->external_kv->physical_blocks();
        }
    }
    for (std::size_t index = 0; status.is_ok() && index < impl_->layers.size(); ++index) {
        if (impl_->external_kv != nullptr) {
            status = impl_->prefill_layer.prefill_sequence_paged(
                &impl_->execution,
                impl_->model->layers[index].decoder_weights(),
                impl_->external_kv->payload(), index, physical_blocks,
                &impl_->prefill_hidden, impl_->position);
        } else {
            status = impl_->prefill_layer.prefill_sequence(
                &impl_->execution,
                impl_->model->layers[index].decoder_weights(),
                &impl_->caches[index], &impl_->prefill_hidden, impl_->position);
        }
    }
    if (status.is_ok()) {
        status = impl_->execution.unpack_rows(impl_->prefill_hidden, row_outputs);
    }
    if (status.is_ok()) {
        status = impl_->execution.rms_norm(
            impl_->prefill_rows.back(), impl_->model->output_norm,
            &impl_->final_hidden, impl_->model->metadata.rms_norm_epsilon);
    }
    if (status.is_ok()) {
        status = impl_->execution.quantized_linear(
            impl_->model->output_weight, impl_->final_hidden, &impl_->logits);
    }
    std::vector<float> result(impl_->model->metadata.vocab_size);
    if (status.is_ok()) {
        status = impl_->execution.download(
            impl_->logits, result.data(), result.size() * sizeof(float));
    }
    if (status.is_ok()) status = impl_->execution.synchronize();
    if (!status.is_ok()) {
        impl_->rollback_caches();
        return status;
    }
    if (impl_->external_kv != nullptr) {
        std::string error;
        if (!impl_->external_kv->commit(impl_->position + tokens.size(), &error)) {
            impl_->rollback_caches();
            return core::Status::error(error);
        }
    }
    impl_->position += tokens.size();
    *logits = std::move(result);
    return core::Status::ok();
}

core::Status RealLlamaCudaContext::copy_layer_kv(
    std::size_t layer,
    std::vector<float>* key,
    std::vector<float>* value) {
    if (layer >= impl_->caches.size() || key == nullptr || value == nullptr) {
        return core::Status::error("CUDA layer KV output is invalid");
    }
    const std::size_t element_count =
        impl_->position * impl_->caches[layer].token_width();
    std::vector<float> key_result(element_count);
    std::vector<float> value_result(element_count);
    core::Status status = impl_->execution.download_kv(
        impl_->caches[layer],
        key_result.data(),
        value_result.data(),
        impl_->position);
    if (status.is_ok()) status = impl_->execution.synchronize();
    if (!status.is_ok()) return status;
    *key = std::move(key_result);
    *value = std::move(value_result);
    return core::Status::ok();
}

void RealLlamaCudaContext::reset() {
    if (impl_->external_kv != nullptr) impl_->external_kv->trim(0);
    for (auto& cache : impl_->caches) cache.clear();
    impl_->position = 0;
}

bool RealLlamaCudaContext::attach_kv_data_plane(
    std::shared_ptr<RealLlamaCudaKvDataPlane> data_plane,
    std::string* error) {
    if (data_plane == nullptr || data_plane->payload() == nullptr) {
        if (error != nullptr) *error = "CUDA KV data plane is invalid";
        return false;
    }
    if (data_plane->token_count() != impl_->position) {
        if (error != nullptr) *error = "CUDA KV data plane position does not match context";
        return false;
    }
    impl_->external_kv = std::move(data_plane);
    for (auto& cache : impl_->caches) cache.clear();
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaCudaContext::adopt_external_kv_prefix(
    std::size_t token_count,
    std::string* error) {
    if (impl_->external_kv == nullptr || impl_->position != 0 || token_count == 0 ||
        token_count >= impl_->capacity_tokens ||
        impl_->external_kv->token_count() != token_count) {
        if (error != nullptr) {
            *error = "CUDA context cannot adopt the external KV prefix";
        }
        return false;
    }
    for (auto& cache : impl_->caches) cache.clear();
    impl_->position = token_count;
    if (error != nullptr) error->clear();
    return true;
}

bool RealLlamaCudaContext::uses_external_kv() const {
    return impl_->external_kv != nullptr;
}

bool RealLlamaCudaContext::uses_external_kv_for(std::size_t request_id) const {
    return impl_->external_kv != nullptr &&
        impl_->external_kv->request_id() == request_id;
}

std::size_t RealLlamaCudaContext::position() const { return impl_->position; }
std::size_t RealLlamaCudaContext::capacity_tokens() const {
    return impl_->capacity_tokens;
}

std::size_t RealLlamaCudaContext::prefill_workspace_creation_count() const {
    return impl_->prefill_workspace_creations;
}

}  // namespace cpp_llm_engine::model
