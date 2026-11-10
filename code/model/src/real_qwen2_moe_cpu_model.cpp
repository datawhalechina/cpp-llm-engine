// Model：实现 Qwen2MoE CPU 全模型前向和请求级分层 K/V。
// 内存策略：全局权重常驻，专家层按层加载并在该层执行后释放。
#include "cpp_llm_engine/model/real_qwen2_moe_cpu_model.h"

#include "chapter05_quantization/q3k_dequantize.h"
#include "cpp_llm_engine/model/real_qwen2_moe_output.h"
#include "gguf_tensor_loader.h"
#include "model_metadata_adapter.h"
#include "real_layer_weights.h"

#include <algorithm>
#include <cstring>
#include <limits>
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
    return info != nullptr && chapter01::GgufTensorLoader::load_tensor(
        model_path, index, *info, output, error);
}

bool lookup_embedding(const chapter01::Tensor& embedding,
                      int token_id,
                      std::vector<float>* output,
                      std::string* error) {
    const auto& shape = embedding.shape();
    if (output == nullptr || shape.size() != 2 || shape[0] <= 0 ||
        shape[1] <= 0 || token_id < 0 ||
        static_cast<std::int64_t>(token_id) >= shape[1]) {
        return fail(error, "Qwen2MoE embedding token is outside vocabulary");
    }
    const auto hidden = static_cast<std::size_t>(shape[0]);
    const auto row = static_cast<std::size_t>(token_id);
    if (embedding.dtype() == chapter01::DType::Float32) {
        const std::size_t row_bytes = hidden * sizeof(float);
        if (embedding.raw_data() == nullptr ||
            row_bytes > embedding.bytes() ||
            row > (embedding.bytes() - row_bytes) / row_bytes) {
            return fail(error, "Qwen2MoE Float32 embedding payload is short");
        }
        std::vector<float> values(hidden);
        std::memcpy(values.data(), embedding.raw_data() + row * row_bytes,
                    row_bytes);
        *output = std::move(values);
        return true;
    }
    if (embedding.dtype() == chapter01::DType::Q3_K) {
        if (hidden % chapter05_quantization::kQ3KBlockElements != 0) {
            return fail(error, "Qwen2MoE Q3_K embedding row is not block aligned");
        }
        const std::size_t row_bytes =
            hidden / chapter05_quantization::kQ3KBlockElements *
            chapter05_quantization::kQ3KBlockBytes;
        if (embedding.raw_data() == nullptr ||
            row_bytes > embedding.bytes() ||
            row > (embedding.bytes() - row_bytes) / row_bytes) {
            return fail(error, "Qwen2MoE Q3_K embedding payload is short");
        }
        return chapter05_quantization::dequantize_q3_k(
            reinterpret_cast<const std::uint8_t*>(embedding.raw_data()) +
                row * row_bytes,
            row_bytes, hidden, output, error);
    }
    return fail(error, "Qwen2MoE embedding format is not supported");
}

std::size_t layer_payload_bytes(
    const chapter01::RealMoeLayerWeights& weights) {
    return weights.attn_norm.bytes() + weights.attn_q.bytes() +
        weights.attn_k.bytes() + weights.attn_v.bytes() +
        weights.attn_q_bias.bytes() + weights.attn_k_bias.bytes() +
        weights.attn_v_bias.bytes() + weights.attn_output.bytes() +
        weights.ffn_norm.bytes() + weights.router.bytes() +
        weights.expert_gate.bytes() + weights.expert_up.bytes() +
        weights.expert_down.bytes() + weights.shared_expert_router.bytes() +
        weights.shared_expert_gate.bytes() + weights.shared_expert_up.bytes() +
        weights.shared_expert_down.bytes();
}

}  // namespace

RealQwen2MoeCpuContext::RealQwen2MoeCpuContext(
    std::shared_ptr<const RealQwen2MoeCpuModel> model)
    : model_(std::move(model)), layers_(model_->metadata_.layer_count) {}

bool RealQwen2MoeCpuContext::prefill_tokens(
    const std::vector<int>& token_ids,
    std::vector<float>* logits,
    std::string* error) {
    if (logits == nullptr || token_ids.empty()) {
        return fail(error, "Qwen2MoE prefill requires tokens and logits output");
    }
    const std::size_t original_position = position();
    if (std::any_of(layers_.begin(), layers_.end(),
                    [original_position](const RealQwen2MoeLayerContext& layer) {
                        return layer.position != original_position;
                    })) {
        return fail(error, "Qwen2MoE layer Context positions diverged");
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
    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        chapter01::RealMoeLayerWeights weights;
        RealQwen2MoePrefillResult layer_result;
        if (!model_->load_layer(layer, &weights, error) ||
            !run_qwen2_moe_layer_with_context(
                model_->metadata_, weights, hidden, &layers_[layer],
                &layer_result, error)) {
            layers_ = saved_layers;
            return false;
        }
        hidden.clear();
        hidden.reserve(layer_result.tokens.size());
        for (auto& token : layer_result.tokens) {
            hidden.push_back(std::move(token.output));
        }
    }

    std::vector<float> normalized;
    std::vector<float> produced_logits;
    if (!finish_qwen2_moe_output(
            model_->metadata_, model_->output_norm_, model_->output_weight_,
            hidden.back(), &normalized, &produced_logits, error)) {
        layers_ = saved_layers;
        return false;
    }
    *logits = std::move(produced_logits);
    if (error != nullptr) error->clear();
    return true;
}

bool RealQwen2MoeCpuContext::decode_token(
    int token_id, std::vector<float>* logits, std::string* error) {
    return prefill_tokens({token_id}, logits, error);
}

void RealQwen2MoeCpuContext::reset() {
    for (auto& layer : layers_) layer.reset();
}

std::size_t RealQwen2MoeCpuContext::position() const {
    return layers_.empty() ? 0 : layers_.front().position;
}

std::size_t RealQwen2MoeCpuContext::layer_count() const {
    return layers_.size();
}

std::shared_ptr<RealQwen2MoeCpuModel> RealQwen2MoeCpuModel::load(
    const std::string& model_path, std::string* error) {
    auto model = std::shared_ptr<RealQwen2MoeCpuModel>(
        new RealQwen2MoeCpuModel());
    model->model_path_ = model_path;
    if (!chapter01::GgufReader::read_index(
            model_path, &model->index_, error) ||
        !chapter01::ModelMetadataAdapter::from_gguf(
            model->index_.metadata, &model->metadata_, error) ||
        !chapter01::ModelSchema::validate_moe_gguf(
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

std::unique_ptr<RealQwen2MoeCpuContext>
RealQwen2MoeCpuModel::create_context() const {
    return std::unique_ptr<RealQwen2MoeCpuContext>(
        new RealQwen2MoeCpuContext(shared_from_this()));
}

bool RealQwen2MoeCpuModel::decode_batch(
    const std::vector<RealQwen2MoeCpuContext*>& contexts,
    const std::vector<int>& token_ids,
    std::vector<std::vector<float>>* logits,
    RealMoeDispatchStats* dispatch_stats,
    std::string* error,
    RealMoeBatchMode batch_mode) const {
    if (logits == nullptr || contexts.empty() ||
        contexts.size() != token_ids.size()) {
        return fail(error, "Qwen2MoE batch Context and token counts differ");
    }
    std::vector<std::vector<RealQwen2MoeLayerContext>> saved;
    std::vector<std::vector<float>> hidden;
    std::set<RealQwen2MoeCpuContext*> unique_contexts;
    saved.reserve(contexts.size());
    hidden.reserve(contexts.size());
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        auto* context = contexts[index];
        if (context == nullptr || !unique_contexts.insert(context).second ||
            context->model_.get() != this ||
            context->layers_.size() != metadata_.layer_count) {
            return fail(error, "Qwen2MoE batch Context is invalid");
        }
        const std::size_t position = context->position();
        if (std::any_of(
                context->layers_.begin(), context->layers_.end(),
                [position](const RealQwen2MoeLayerContext& layer) {
                    return layer.position != position;
                })) {
            return fail(error, "Qwen2MoE batch layer positions diverged");
        }
        std::vector<float> embedding;
        if (!lookup_embedding(embedding_, token_ids[index], &embedding, error)) {
            return false;
        }
        saved.push_back(context->layers_);
        hidden.push_back(std::move(embedding));
    }

    RealMoeDispatchStats total_stats;
    for (std::size_t layer = 0; layer < metadata_.layer_count; ++layer) {
        chapter01::RealMoeLayerWeights weights;
        if (!load_layer(layer, &weights, error)) {
            for (std::size_t index = 0; index < contexts.size(); ++index) {
                contexts[index]->layers_ = std::move(saved[index]);
            }
            return false;
        }
        std::vector<RealQwen2MoeLayerContext*> layer_contexts;
        layer_contexts.reserve(contexts.size());
        for (auto* context : contexts) {
            layer_contexts.push_back(&context->layers_[layer]);
        }
        std::vector<RealQwen2MoeLayerResult> layer_results;
        RealMoeDispatchStats layer_stats;
        if (!run_qwen2_moe_layer_batch_with_context(
                metadata_, weights, hidden, layer_contexts, &layer_results,
                &layer_stats, error, {}, batch_mode)) {
            for (std::size_t index = 0; index < contexts.size(); ++index) {
                contexts[index]->layers_ = std::move(saved[index]);
            }
            return false;
        }
        total_stats.assignment_count += layer_stats.assignment_count;
        total_stats.active_expert_count += layer_stats.active_expert_count;
        total_stats.maximum_tokens_per_expert = std::max(
            total_stats.maximum_tokens_per_expert,
            layer_stats.maximum_tokens_per_expert);
        for (std::size_t index = 0; index < hidden.size(); ++index) {
            hidden[index] = std::move(layer_results[index].output);
        }
    }

    std::vector<std::vector<float>> built(contexts.size());
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        std::vector<float> normalized;
        if (!finish_qwen2_moe_output(
                metadata_, output_norm_, output_weight_, hidden[index],
                &normalized, &built[index], error)) {
            for (std::size_t restore = 0; restore < contexts.size(); ++restore) {
                contexts[restore]->layers_ = std::move(saved[restore]);
            }
            return false;
        }
    }
    *logits = std::move(built);
    if (dispatch_stats != nullptr) *dispatch_stats = total_stats;
    if (error != nullptr) error->clear();
    return true;
}

const chapter01::ModelMetadata& RealQwen2MoeCpuModel::metadata() const {
    return metadata_;
}

void RealQwen2MoeCpuModel::set_layer_cache_capacity(
    std::size_t layer_count) {
    std::lock_guard<std::mutex> lock(layer_cache_mutex_);
    layer_cache_capacity_ = std::min(layer_count, metadata_.layer_count);
    layer_cache_.clear();
    layer_cache_clock_ = 0;
    layer_cache_hits_ = 0;
    layer_cache_misses_ = 0;
}

RealQwen2MoeLayerCacheStats RealQwen2MoeCpuModel::layer_cache_stats() const {
    std::lock_guard<std::mutex> lock(layer_cache_mutex_);
    RealQwen2MoeLayerCacheStats stats;
    stats.capacity = layer_cache_capacity_;
    stats.resident_layers = layer_cache_.size();
    stats.hits = layer_cache_hits_;
    stats.misses = layer_cache_misses_;
    for (const auto& layer : layer_cache_) stats.resident_bytes += layer.bytes;
    return stats;
}

bool RealQwen2MoeCpuModel::load_layer(
    std::size_t layer,
    chapter01::RealMoeLayerWeights* output,
    std::string* error) const {
    if (output == nullptr || layer >= metadata_.layer_count) {
        return fail(error, "Qwen2MoE layer cache request is invalid");
    }
    {
        std::lock_guard<std::mutex> lock(layer_cache_mutex_);
        const auto found = std::find_if(
            layer_cache_.begin(), layer_cache_.end(),
            [layer](const CachedLayer& cached) {
                return cached.index == layer;
            });
        if (found != layer_cache_.end()) {
            found->last_use = ++layer_cache_clock_;
            ++layer_cache_hits_;
            *output = found->weights;
            if (error != nullptr) error->clear();
            return true;
        }
        ++layer_cache_misses_;
    }

    chapter01::RealMoeLayerWeights loaded;
    if (!chapter01::RealMoeLayerLoader::load(
            model_path_, index_, layout_, layer, &loaded, error)) {
        return false;
    }
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
                *output = raced->weights;
                if (error != nullptr) error->clear();
                return true;
            }
            if (layer_cache_.size() == layer_cache_capacity_) {
                const auto oldest = std::min_element(
                    layer_cache_.begin(), layer_cache_.end(),
                    [](const CachedLayer& left, const CachedLayer& right) {
                        return left.last_use < right.last_use;
                    });
                layer_cache_.erase(oldest);
            }
            layer_cache_.push_back(
                {layer, layer_payload_bytes(loaded), ++layer_cache_clock_,
                 loaded});
        }
    }
    *output = std::move(loaded);
    if (error != nullptr) error->clear();
    return true;
}

}  // namespace cpp_llm_engine::model
