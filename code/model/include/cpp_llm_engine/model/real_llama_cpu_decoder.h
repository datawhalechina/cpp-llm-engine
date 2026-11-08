// 真实模型执行层：声明 Llama GGUF CPU 模型与请求执行上下文。
// 职责：Model 共享只读量化权重，Context 独占 position、分层 KV 和执行统计。
// 验证：多层数值对照与 Engine golden CTest 检查结果和 context 隔离。
#pragma once

#include "chapter05_quantization/quantized_linear_policy.h"
#include "cpp_llm_engine/core/decoder_contract.h"
#include "tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// 暂时保留 chapter01 命名空间，兼容已有 Engine 和多模态调用方；
// 文件与构建目标已经归入独立 model 模块。
namespace chapter01 {

struct RealLlamaDecoderStats {
    std::size_t model_layer_count = 0;
    std::size_t loaded_layer_count = 0;
    std::size_t token_count = 0;
    std::size_t query_head_count = 0;
    std::size_t kv_head_count = 0;
    std::size_t head_dimension = 0;
    std::size_t rope_dimension = 0;
    std::size_t attention_score_count = 0;
    float rope_frequency_base = 0.0f;
};

struct RealLlamaDecoderProfile {
    std::size_t forward_tokens = 0;
    double total_ms = 0.0;
    double embedding_ms = 0.0;
    double normalization_ms = 0.0;
    double qkv_projection_ms = 0.0;
    double rope_ms = 0.0;
    double attention_ms = 0.0;
    double attention_output_ms = 0.0;
    double ffn_ms = 0.0;
    double ffn_gate_ms = 0.0;
    double ffn_up_ms = 0.0;
    double ffn_activation_ms = 0.0;
    double ffn_down_ms = 0.0;
    double ffn_residual_ms = 0.0;
    double lm_head_ms = 0.0;
    std::size_t batch_linear_calls = 0;
    double batch_input_pack_ms = 0.0;
    double batch_kernel_ms = 0.0;
    double batch_output_unpack_ms = 0.0;
};

class RealLlamaCpuContext;

class RealLlamaKvDataPlane {
public:
    virtual ~RealLlamaKvDataPlane() = default;

    virtual std::size_t token_count() const = 0;
    virtual bool reserve(std::size_t total_tokens, std::string* error) = 0;
    virtual bool write_token(
        std::size_t layer,
        std::size_t token_index,
        const float* key,
        const float* value,
        std::size_t element_count,
        std::string* error) = 0;
    virtual const float* key_data(
        std::size_t layer,
        std::size_t token_index,
        std::size_t kv_head) const = 0;
    virtual const float* value_data(
        std::size_t layer,
        std::size_t token_index,
        std::size_t kv_head) const = 0;
    virtual bool commit(std::size_t total_tokens, std::string* error) = 0;
    virtual bool trim(std::size_t total_tokens) = 0;
};

struct RealLlamaCpuBatchItem {
    RealLlamaCpuContext* context = nullptr;
    TokenId token = 0;
};

class RealLlamaCpuModel final {
public:
    static std::unique_ptr<RealLlamaCpuModel> load(
        const std::string& model_path,
        std::size_t layer_count = 0,
        std::string* error = nullptr);

    ~RealLlamaCpuModel();
    RealLlamaCpuModel(RealLlamaCpuModel&&) noexcept;
    RealLlamaCpuModel& operator=(RealLlamaCpuModel&&) noexcept;

    RealLlamaCpuModel(const RealLlamaCpuModel&) = delete;
    RealLlamaCpuModel& operator=(const RealLlamaCpuModel&) = delete;

    std::unique_ptr<RealLlamaCpuContext> create_context() const;
    std::size_t layer_count() const;
    std::size_t model_layer_count() const;
    std::size_t vocab_size() const;
    TokenId eos_token_id() const;
    std::size_t weight_payload_bytes() const;
    cpp_llm_engine::core::DecoderModelContract decoder_contract() const;
    bool embed_tokens(
        const TokenSequence& tokens,
        cpp_llm_engine::core::DecoderEmbeddingSequence* embeddings,
        std::string* error = nullptr) const;
    bool decode_batch(
        const std::vector<RealLlamaCpuBatchItem>& items,
        std::vector<std::vector<float>>* logits,
        std::string* error = nullptr) const;

private:
    struct Impl;
    explicit RealLlamaCpuModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class RealLlamaCpuContext final : public cpp_llm_engine::core::DecoderExecutionContext {
public:
    ~RealLlamaCpuContext() override;
    RealLlamaCpuContext(RealLlamaCpuContext&&) noexcept;
    RealLlamaCpuContext& operator=(RealLlamaCpuContext&&) noexcept;

    RealLlamaCpuContext(const RealLlamaCpuContext&) = delete;
    RealLlamaCpuContext& operator=(const RealLlamaCpuContext&) = delete;

    void reset() override;
    bool prefill(const TokenSequence& tokens,
                 std::vector<float>* logits,
                 std::string* error = nullptr);
    bool decode(TokenId token,
                std::vector<float>* logits,
                std::string* error = nullptr);
    bool forward_hidden_state(
        const std::vector<float>& input,
        std::vector<float>* output,
        std::string* error = nullptr);
    cpp_llm_engine::core::Status prefill_tokens(
        const cpp_llm_engine::core::DecoderTokenSequence& tokens,
        cpp_llm_engine::core::DecoderLogits* logits) override;
    cpp_llm_engine::core::Status prefill_chunk(
        const cpp_llm_engine::core::DecoderTokenSequence& tokens,
        bool start_new_sequence,
        bool produce_logits,
        cpp_llm_engine::core::DecoderLogits* logits) override;
    cpp_llm_engine::core::Status prefill_embeddings(
        const cpp_llm_engine::core::DecoderEmbeddingSequence& embeddings,
        cpp_llm_engine::core::DecoderLogits* logits) override;
    cpp_llm_engine::core::Status decode_token(
        cpp_llm_engine::core::DecoderTokenId token,
        cpp_llm_engine::core::DecoderLogits* logits) override;
    std::size_t token_count() const override;
    std::size_t kv_cache_bytes() const;
    bool copy_kv_token(
        std::size_t layer,
        std::size_t token_index,
        std::vector<float>* key,
        std::vector<float>* value,
        std::string* error = nullptr) const;
    bool attach_kv_data_plane(
        std::shared_ptr<RealLlamaKvDataPlane> data_plane,
        std::string* error = nullptr);
    bool adopt_external_kv_prefix(
        std::size_t token_count,
        std::string* error = nullptr);
    bool uses_external_kv() const;
    cpp_llm_engine::core::DecoderModelContract model_contract() const override;

    void set_profiling_enabled(bool enabled);
    bool set_attention_linear_policy(
        const chapter05_quantization::QuantizedLinearPolicy& policy,
        std::string* error = nullptr);
    bool set_ffn_linear_policy(
        const chapter05_quantization::QuantizedLinearPolicy& policy,
        std::string* error = nullptr);
    RealLlamaDecoderStats stats() const;
    RealLlamaDecoderProfile profile() const;

private:
    friend class RealLlamaCpuModel;
    struct Impl;
    explicit RealLlamaCpuContext(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace chapter01
