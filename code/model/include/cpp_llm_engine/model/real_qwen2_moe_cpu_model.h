// Model：声明 Qwen2MoE CPU 模型与每请求 Context。
// 所有权：Model 共享只读 GGUF 索引和全局权重；Context 独占分层 K/V。
// 契约：成功时所有层推进相同词元数，失败时整组分层状态保持不变。
#pragma once

#include "cpp_llm_engine/model/real_qwen2_moe_layer.h"
#include "gguf_reader.h"
#include "model_schema.h"
#include "tensor.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cpp_llm_engine::model {

class RealQwen2MoeCpuModel;

struct RealQwen2MoeLayerCacheStats {
    std::size_t capacity = 0;
    std::size_t resident_layers = 0;
    std::size_t resident_bytes = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
};

class RealQwen2MoeCpuContext final {
public:
    bool prefill_tokens(const std::vector<int>& token_ids,
                        std::vector<float>* logits,
                        std::string* error = nullptr);
    bool decode_token(int token_id,
                      std::vector<float>* logits,
                      std::string* error = nullptr);
    void reset();

    std::size_t position() const;
    std::size_t layer_count() const;

private:
    friend class RealQwen2MoeCpuModel;
    explicit RealQwen2MoeCpuContext(
        std::shared_ptr<const RealQwen2MoeCpuModel> model);

    std::shared_ptr<const RealQwen2MoeCpuModel> model_;
    std::vector<RealQwen2MoeLayerContext> layers_;
};

class RealQwen2MoeCpuModel final
    : public std::enable_shared_from_this<RealQwen2MoeCpuModel> {
public:
    static std::shared_ptr<RealQwen2MoeCpuModel> load(
        const std::string& model_path,
        std::string* error = nullptr);

    std::unique_ptr<RealQwen2MoeCpuContext> create_context() const;
    bool decode_batch(
        const std::vector<RealQwen2MoeCpuContext*>& contexts,
        const std::vector<int>& token_ids,
        std::vector<std::vector<float>>* logits,
        RealMoeDispatchStats* dispatch_stats = nullptr,
        std::string* error = nullptr,
        RealMoeBatchMode batch_mode = RealMoeBatchMode::Grouped) const;
    const chapter01::ModelMetadata& metadata() const;
    void set_layer_cache_capacity(std::size_t layer_count);
    RealQwen2MoeLayerCacheStats layer_cache_stats() const;

private:
    friend class RealQwen2MoeCpuContext;

    struct CachedLayer {
        std::size_t index = 0;
        std::size_t bytes = 0;
        std::size_t last_use = 0;
        chapter01::RealMoeLayerWeights weights;
    };

    bool load_layer(std::size_t layer,
                    chapter01::RealMoeLayerWeights* output,
                    std::string* error) const;

    std::string model_path_;
    chapter01::GgufModelIndex index_;
    chapter01::ModelMetadata metadata_;
    chapter01::MoeDecoderTensorLayout layout_;
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
