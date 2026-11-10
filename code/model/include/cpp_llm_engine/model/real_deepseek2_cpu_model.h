// Model：声明 DeepSeek2 CPU 模型与请求级分层 Context。
// 所有权：Model 共享 GGUF 索引和全局权重；Context 独占每层 MLA 历史。
// 契约：成功时全部层推进相同位置，失败时恢复整组 Context。
#pragma once

#include "cpp_llm_engine/model/real_deepseek2_dense_layer.h"
#include "gguf_reader.h"
#include "model_schema.h"
#include "tensor.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cpp_llm_engine::model {

class RealDeepSeek2CpuModel;

struct RealDeepSeek2LayerCacheStats {
    std::size_t capacity = 0;
    std::size_t resident_layers = 0;
    std::size_t resident_bytes = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
};

class RealDeepSeek2CpuContext final {
public:
    bool prefill_tokens(const std::vector<int>& token_ids,
                        std::vector<float>* logits,
                        std::string* error = nullptr);
    bool decode_token(int token_id,
                      std::vector<float>* logits,
                      std::string* error = nullptr);
    bool forward_hidden_state(const std::vector<float>& input,
                              std::vector<float>* output,
                              std::string* error = nullptr);
    void reset();

    std::size_t position() const;
    std::size_t layer_count() const;

private:
    friend class RealDeepSeek2CpuModel;
    explicit RealDeepSeek2CpuContext(
        std::shared_ptr<const RealDeepSeek2CpuModel> model);

    bool forward_hidden(const std::vector<float>& input,
                        std::vector<float>* output,
                        std::string* error);

    std::shared_ptr<const RealDeepSeek2CpuModel> model_;
    std::vector<RealDeepSeek2LayerContext> layers_;
};

class RealDeepSeek2CpuModel final
    : public std::enable_shared_from_this<RealDeepSeek2CpuModel> {
public:
    static std::shared_ptr<RealDeepSeek2CpuModel> load(
        const std::string& model_path,
        std::string* error = nullptr);

    std::unique_ptr<RealDeepSeek2CpuContext> create_context() const;
    bool decode_batch(
        const std::vector<RealDeepSeek2CpuContext*>& contexts,
        const std::vector<int>& token_ids,
        std::vector<std::vector<float>>* logits,
        std::string* error = nullptr,
        RealMoeBatchMode mode = RealMoeBatchMode::Grouped) const;
    const chapter01::ModelMetadata& metadata() const;
    void set_layer_cache_capacity(std::size_t layer_count);
    RealDeepSeek2LayerCacheStats layer_cache_stats() const;

private:
    friend class RealDeepSeek2CpuContext;

    struct CachedLayer {
        std::size_t index = 0;
        std::size_t bytes = 0;
        std::size_t last_use = 0;
        chapter01::RealDeepSeek2DenseLayerWeights dense;
        chapter01::RealDeepSeek2MoeLayerWeights moe;
    };

    bool load_dense_layer(chapter01::RealDeepSeek2DenseLayerWeights* output,
                          std::string* error) const;
    bool load_moe_layer(std::size_t layer,
                        chapter01::RealDeepSeek2MoeLayerWeights* output,
                        std::string* error) const;

    std::string model_path_;
    chapter01::GgufModelIndex index_;
    chapter01::ModelMetadata metadata_;
    chapter01::DeepSeek2DecoderTensorLayout layout_;
    chapter01::Tensor embedding_;
    chapter01::Tensor output_norm_;
    chapter01::Tensor output_weight_;
    mutable std::mutex layer_cache_mutex_;
    mutable std::vector<CachedLayer> layer_cache_;
    mutable std::size_t layer_cache_clock_ = 0;
    mutable std::size_t layer_cache_hits_ = 0;
    mutable std::size_t layer_cache_misses_ = 0;
    std::size_t layer_cache_capacity_ = 0;
};

}  // namespace cpp_llm_engine::model
