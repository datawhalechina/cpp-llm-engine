// Model：实现 DeepSeek2 CPU 全模型前向和请求级分层 MLA Context。
#include "cpp_llm_engine/model/real_deepseek2_cpu_model.h"

#include "chapter05_quantization/q2k_dequantize.h"
#include "cpp_llm_engine/model/real_qwen2_moe_output.h"
#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "real_layer_weights.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>

namespace cpp_llm_engine::model {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

const chapter01::GgufTensorInfo* find_tensor(
    const chapter01::GgufModelIndex& index,
    const std::string& name) {
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
    if (info == nullptr) return fail(error, "DeepSeek2 tensor is missing: " + name);
    return chapter01::GgufTensorLoader::load_tensor(
        model_path, index, *info, output, error);
}

bool lookup_embedding(const chapter01::Tensor& embedding,
                      int token_id,
                      std::vector<float>* output,
                      std::string* error) {
    const auto& shape = embedding.shape();
    if (output == nullptr || shape.size() != 2 || shape[0] <= 0 ||
        shape[1] <= 0 || token_id < 0 ||
        static_cast<std::int64_t>(token_id) >= shape[1] ||
        embedding.dtype() != chapter01::DType::Q2_K) {
        return fail(error, "DeepSeek2 embedding request is invalid");
    }
    const std::size_t hidden = static_cast<std::size_t>(shape[0]);
    if (hidden % chapter05_quantization::kQ2KBlockElements != 0) {
        return fail(error, "DeepSeek2 embedding row is not Q2_K aligned");
    }
    const std::size_t row_bytes =
        hidden / chapter05_quantization::kQ2KBlockElements *
        chapter05_quantization::kQ2KBlockBytes;
    const std::size_t row = static_cast<std::size_t>(token_id);
    if (embedding.raw_data() == nullptr || row_bytes > embedding.bytes() ||
        row > (embedding.bytes() - row_bytes) / row_bytes) {
        return fail(error, "DeepSeek2 embedding payload is short");
    }
    return chapter05_quantization::dequantize_q2_k(
        reinterpret_cast<const std::uint8_t*>(embedding.raw_data()) +
            row * row_bytes,
        row_bytes, hidden, output, error);
}

std::size_t payload_bytes(
    const chapter01::RealDeepSeek2DenseLayerWeights& w) {
    return w.attn_norm.bytes() + w.attn_q.bytes() + w.attn_kv_a_mqa.bytes() +
        w.attn_kv_a_norm.bytes() + w.attn_kv_b.bytes() + w.attn_k_b.bytes() +
        w.attn_v_b.bytes() + w.attn_output.bytes() + w.ffn_norm.bytes() +
        w.ffn_gate.bytes() + w.ffn_up.bytes() + w.ffn_down.bytes();
}

std::size_t payload_bytes(const chapter01::RealDeepSeek2MoeLayerWeights& w) {
    return w.attn_norm.bytes() + w.attn_q.bytes() + w.attn_kv_a_mqa.bytes() +
        w.attn_kv_a_norm.bytes() + w.attn_kv_b.bytes() + w.attn_k_b.bytes() +
        w.attn_v_b.bytes() + w.attn_output.bytes() + w.ffn_norm.bytes() +
        w.router.bytes() + w.expert_gate.bytes() + w.expert_up.bytes() +
        w.expert_down.bytes() + w.shared_expert_gate.bytes() +
        w.shared_expert_up.bytes() + w.shared_expert_down.bytes();
}

}  // namespace

RealDeepSeek2CpuContext::RealDeepSeek2CpuContext(
    std::shared_ptr<const RealDeepSeek2CpuModel> model)
    : model_(std::move(model)), layers_(model_->metadata_.layer_count) {}

bool RealDeepSeek2CpuContext::forward_hidden(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::string* error) {
    if (output == nullptr || input.size() != model_->metadata_.hidden_size ||
        layers_.size() != model_->metadata_.layer_count || layers_.empty()) {
        return fail(error, "DeepSeek2 hidden forward input is invalid");
    }
    const std::size_t original_position = position();
    if (std::any_of(layers_.begin(), layers_.end(),
                    [original_position](const RealDeepSeek2LayerContext& layer) {
                        return layer.position != original_position;
                    })) {
        return fail(error, "DeepSeek2 layer Context positions diverged");
    }

    const auto saved_layers = layers_;
    std::vector<float> hidden = input;
    chapter01::RealDeepSeek2DenseLayerWeights dense;
    RealDeepSeek2LayerResult layer_result;
    if (!model_->load_dense_layer(&dense, error) ||
        !run_deepseek2_dense_token_layer(
            model_->metadata_, dense, hidden, &layers_[0], &layer_result,
            error)) {
        layers_ = saved_layers;
        return false;
    }
    hidden = std::move(layer_result.output);

    for (std::size_t layer = 1; layer < layers_.size(); ++layer) {
        chapter01::RealDeepSeek2MoeLayerWeights weights;
        RealMoeRoute route;
        if (!model_->load_moe_layer(layer, &weights, error) ||
            !run_deepseek2_moe_token_layer(
                model_->metadata_, weights, hidden, &layers_[layer],
                &layer_result, &route, error)) {
            layers_ = saved_layers;
            return false;
        }
        hidden = std::move(layer_result.output);
    }
    *output = std::move(hidden);
    if (error != nullptr) error->clear();
    return true;
}

bool RealDeepSeek2CpuContext::forward_hidden_state(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::string* error) {
    return forward_hidden(input, output, error);
}

bool RealDeepSeek2CpuContext::prefill_tokens(
    const std::vector<int>& token_ids,
    std::vector<float>* logits,
    std::string* error) {
    if (logits == nullptr || token_ids.empty()) {
        return fail(error, "DeepSeek2 prefill requires tokens and logits output");
    }
    const std::size_t original_position = position();
    if (std::any_of(layers_.begin(), layers_.end(),
                    [original_position](const RealDeepSeek2LayerContext& layer) {
                        return layer.position != original_position;
                    })) {
        return fail(error, "DeepSeek2 layer Context positions diverged");
    }

    std::vector<std::vector<float>> hidden;
    hidden.reserve(token_ids.size());
    for (const int token_id : token_ids) {
        std::vector<float> embedding;
        if (!lookup_embedding(model_->embedding_, token_id, &embedding, error)) {
            return false;
        }
        hidden.push_back(std::move(embedding));
    }

    const auto saved_layers = layers_;
    chapter01::RealDeepSeek2DenseLayerWeights dense;
    if (!model_->load_dense_layer(&dense, error)) return false;
    for (auto& token : hidden) {
        RealDeepSeek2LayerResult result;
        if (!run_deepseek2_dense_token_layer(
                model_->metadata_, dense, token, &layers_[0], &result,
                error)) {
            layers_ = saved_layers;
            return false;
        }
        token = std::move(result.output);
    }

    for (std::size_t layer = 1; layer < layers_.size(); ++layer) {
        chapter01::RealDeepSeek2MoeLayerWeights weights;
        if (!model_->load_moe_layer(layer, &weights, error)) {
            layers_ = saved_layers;
            return false;
        }
        for (auto& token : hidden) {
            RealDeepSeek2LayerResult result;
            RealMoeRoute route;
            if (!run_deepseek2_moe_token_layer(
                    model_->metadata_, weights, token, &layers_[layer],
                    &result, &route, error)) {
                layers_ = saved_layers;
                return false;
            }
            token = std::move(result.output);
        }
    }
    std::vector<float> normalized;
    std::vector<float> produced_logits;
    if (!finish_quantized_model_output(
            model_->metadata_, model_->output_norm_, model_->output_weight_,
            hidden.back(), &normalized, &produced_logits, error)) {
        layers_ = saved_layers;
        return false;
    }
    *logits = std::move(produced_logits);
    if (error != nullptr) error->clear();
    return true;
}

bool RealDeepSeek2CpuContext::decode_token(
    int token_id, std::vector<float>* logits, std::string* error) {
    return prefill_tokens({token_id}, logits, error);
}

void RealDeepSeek2CpuContext::reset() {
    for (auto& layer : layers_) layer.reset();
}

std::size_t RealDeepSeek2CpuContext::position() const {
    return layers_.empty() ? 0 : layers_.front().position;
}

std::size_t RealDeepSeek2CpuContext::layer_count() const {
    return layers_.size();
}

std::shared_ptr<RealDeepSeek2CpuModel> RealDeepSeek2CpuModel::load(
    const std::string& model_path, std::string* error) {
    auto model = std::shared_ptr<RealDeepSeek2CpuModel>(
        new RealDeepSeek2CpuModel());
    model->model_path_ = model_path;
    if (!chapter01::GgufReader::read_index(
            model_path, &model->index_, error) ||
        !chapter01::ModelMetadataAdapter::from_gguf(
            model->index_.metadata, &model->metadata_, error) ||
        !chapter01::ModelSchema::validate_deepseek2_gguf(
            model->metadata_, model->index_, &model->layout_, error) ||
        !load_named(model_path, model->index_, model->layout_.embedding,
                    &model->embedding_, error) ||
        !load_named(model_path, model->index_, model->layout_.output_norm,
                    &model->output_norm_, error) ||
        !load_named(model_path, model->index_, model->layout_.output,
                    &model->output_weight_, error)) {
        return nullptr;
    }
    if (error != nullptr) error->clear();
    return model;
}

std::unique_ptr<RealDeepSeek2CpuContext>
RealDeepSeek2CpuModel::create_context() const {
    return std::unique_ptr<RealDeepSeek2CpuContext>(
        new RealDeepSeek2CpuContext(shared_from_this()));
}

bool RealDeepSeek2CpuModel::decode_batch(
    const std::vector<RealDeepSeek2CpuContext*>& contexts,
    const std::vector<int>& token_ids,
    std::vector<std::vector<float>>* logits,
    std::string* error,
    RealMoeBatchMode mode) const {
    if (logits == nullptr || contexts.empty() ||
        contexts.size() != token_ids.size()) {
        return fail(error, "DeepSeek2 batch Context and token counts differ");
    }

    std::set<RealDeepSeek2CpuContext*> unique_contexts;
    std::vector<std::vector<RealDeepSeek2LayerContext>> saved;
    std::vector<std::vector<float>> hidden;
    saved.reserve(contexts.size());
    hidden.reserve(contexts.size());
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        auto* context = contexts[index];
        if (context == nullptr || !unique_contexts.insert(context).second ||
            context->model_.get() != this ||
            context->layers_.size() != metadata_.layer_count) {
            return fail(error, "DeepSeek2 batch Context is invalid");
        }
        const std::size_t position = context->position();
        if (std::any_of(
                context->layers_.begin(), context->layers_.end(),
                [position](const RealDeepSeek2LayerContext& layer) {
                    return layer.position != position;
                })) {
            return fail(error, "DeepSeek2 batch layer positions diverged");
        }
        std::vector<float> embedding;
        if (!lookup_embedding(embedding_, token_ids[index], &embedding, error)) {
            return false;
        }
        saved.push_back(context->layers_);
        hidden.push_back(std::move(embedding));
    }
    const auto rollback = [&]() {
        for (std::size_t index = 0; index < contexts.size(); ++index) {
            contexts[index]->layers_ = saved[index];
        }
    };

    chapter01::RealDeepSeek2DenseLayerWeights dense;
    if (!load_dense_layer(&dense, error)) return false;
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        RealDeepSeek2LayerResult result;
        if (!run_deepseek2_dense_token_layer(
                metadata_, dense, hidden[index], &contexts[index]->layers_[0],
                &result, error)) {
            rollback();
            return false;
        }
        hidden[index] = std::move(result.output);
    }

    for (std::size_t layer = 1; layer < metadata_.layer_count; ++layer) {
        chapter01::RealDeepSeek2MoeLayerWeights weights;
        if (!load_moe_layer(layer, &weights, error)) {
            rollback();
            return false;
        }
        std::vector<RealDeepSeek2LayerContext*> layer_contexts;
        layer_contexts.reserve(contexts.size());
        for (auto* context : contexts) {
            layer_contexts.push_back(&context->layers_[layer]);
        }
        std::vector<RealDeepSeek2LayerResult> results;
        std::vector<RealMoeRoute> routes;
        RealMoeDispatchStats stats;
        if (!run_deepseek2_moe_token_layer_batch(
                metadata_, weights, hidden, layer_contexts, &results,
                &routes, &stats, error, {}, mode)) {
            rollback();
            return false;
        }
        for (std::size_t index = 0; index < contexts.size(); ++index) {
            hidden[index] = std::move(results[index].output);
        }
    }

    std::vector<std::vector<float>> built(contexts.size());
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        std::vector<float> normalized;
        if (!finish_quantized_model_output(
                metadata_, output_norm_, output_weight_, hidden[index],
                &normalized, &built[index], error)) {
            rollback();
            return false;
        }
    }
    *logits = std::move(built);
    if (error != nullptr) error->clear();
    return true;
}

const chapter01::ModelMetadata& RealDeepSeek2CpuModel::metadata() const {
    return metadata_;
}

void RealDeepSeek2CpuModel::set_layer_cache_capacity(std::size_t layer_count) {
    std::lock_guard<std::mutex> lock(layer_cache_mutex_);
    layer_cache_capacity_ = std::min(layer_count, metadata_.layer_count);
    layer_cache_.clear();
    layer_cache_clock_ = 0;
    layer_cache_hits_ = 0;
    layer_cache_misses_ = 0;
}

RealDeepSeek2LayerCacheStats RealDeepSeek2CpuModel::layer_cache_stats() const {
    std::lock_guard<std::mutex> lock(layer_cache_mutex_);
    RealDeepSeek2LayerCacheStats stats;
    stats.capacity = layer_cache_capacity_;
    stats.resident_layers = layer_cache_.size();
    stats.hits = layer_cache_hits_;
    stats.misses = layer_cache_misses_;
    for (const auto& layer : layer_cache_) stats.resident_bytes += layer.bytes;
    return stats;
}

bool RealDeepSeek2CpuModel::load_dense_layer(
    chapter01::RealDeepSeek2DenseLayerWeights* output,
    std::string* error) const {
    if (output == nullptr) return fail(error, "DeepSeek2 dense cache output is null");
    {
        std::lock_guard<std::mutex> lock(layer_cache_mutex_);
        const auto found = std::find_if(
            layer_cache_.begin(), layer_cache_.end(),
            [](const CachedLayer& cached) { return cached.index == 0; });
        if (found != layer_cache_.end()) {
            found->last_use = ++layer_cache_clock_;
            ++layer_cache_hits_;
            *output = found->dense;
            return true;
        }
        ++layer_cache_misses_;
    }
    chapter01::RealDeepSeek2DenseLayerWeights loaded;
    if (!chapter01::RealDeepSeek2DenseLayerLoader::load(
            model_path_, index_, layout_, 0, &loaded, error)) return false;
    {
        std::lock_guard<std::mutex> lock(layer_cache_mutex_);
        if (layer_cache_capacity_ > 0) {
            const auto raced = std::find_if(
                layer_cache_.begin(), layer_cache_.end(),
                [](const CachedLayer& cached) { return cached.index == 0; });
            if (raced != layer_cache_.end()) {
                raced->last_use = ++layer_cache_clock_;
                *output = raced->dense;
                return true;
            }
            if (layer_cache_.size() == layer_cache_capacity_) {
                layer_cache_.erase(std::min_element(
                    layer_cache_.begin(), layer_cache_.end(),
                    [](const CachedLayer& a, const CachedLayer& b) {
                        return a.last_use < b.last_use;
                    }));
            }
            layer_cache_.push_back(
                {0, payload_bytes(loaded), ++layer_cache_clock_, loaded, {}});
        }
    }
    *output = std::move(loaded);
    return true;
}

bool RealDeepSeek2CpuModel::load_moe_layer(
    std::size_t layer,
    chapter01::RealDeepSeek2MoeLayerWeights* output,
    std::string* error) const {
    if (output == nullptr || layer == 0 || layer >= metadata_.layer_count) {
        return fail(error, "DeepSeek2 MoE cache request is invalid");
    }
    {
        std::lock_guard<std::mutex> lock(layer_cache_mutex_);
        const auto found = std::find_if(
            layer_cache_.begin(), layer_cache_.end(),
            [layer](const CachedLayer& cached) { return cached.index == layer; });
        if (found != layer_cache_.end()) {
            found->last_use = ++layer_cache_clock_;
            ++layer_cache_hits_;
            *output = found->moe;
            return true;
        }
        ++layer_cache_misses_;
    }
    chapter01::RealDeepSeek2MoeLayerWeights loaded;
    if (!chapter01::RealDeepSeek2MoeLayerLoader::load(
            model_path_, index_, layout_, layer, &loaded, error)) return false;
    {
        std::lock_guard<std::mutex> lock(layer_cache_mutex_);
        if (layer_cache_capacity_ > 0) {
            const auto raced = std::find_if(
                layer_cache_.begin(), layer_cache_.end(),
                [layer](const CachedLayer& cached) {
                    return cached.index == layer;
                });
            if (raced != layer_cache_.end()) {
                raced->last_use = ++layer_cache_clock_;
                *output = raced->moe;
                return true;
            }
            if (layer_cache_.size() == layer_cache_capacity_) {
                layer_cache_.erase(std::min_element(
                    layer_cache_.begin(), layer_cache_.end(),
                    [](const CachedLayer& a, const CachedLayer& b) {
                        return a.last_use < b.last_use;
                    }));
            }
            layer_cache_.push_back(
                {layer, payload_bytes(loaded), ++layer_cache_clock_, {}, loaded});
        }
    }
    *output = std::move(loaded);
    return true;
}

}  // namespace cpp_llm_engine::model
