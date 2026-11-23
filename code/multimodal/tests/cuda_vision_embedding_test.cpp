// 多模态 CUDA：验证 Patch/Position Embedding 与 CPU reference 一致。
#include "multimodal/cuda_vision_embedding.h"

#include "cpp_llm_engine/cuda/cuda_execution_context.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed: " #condition << " at line " << __LINE__ \
                      << '\n';                                                  \
            return 1;                                                           \
        }                                                                       \
    } while (false)

int main() {
    using cpp_llm_engine::core::DType;
    using cpp_llm_engine::cuda::CudaExecutionContext;
    using cpp_llm_engine::cuda::CudaTensor;

    const multimodal::PatchTensor patches{
        3, 4,
        {0.25F, -0.5F, 0.75F, 1.0F,
         -1.0F, 0.5F, 0.25F, -0.75F,
         0.6F, 0.4F, -0.2F, 0.8F}};
    const multimodal::PatchEmbeddingWeights weights{
        4, 3, 5,
        {0.1F, 0.2F, 0.3F, 0.4F,
         -0.5F, 0.6F, -0.7F, 0.8F,
         0.9F, -1.0F, 1.1F, -1.2F},
        {0.01F, -0.02F, 0.03F},
        {0.00F, 0.01F, 0.02F,
         0.03F, 0.04F, 0.05F,
         0.06F, 0.07F, 0.08F,
         0.09F, 0.10F, 0.11F,
         0.12F, 0.13F, 0.14F}};
    const std::vector<std::size_t> positions{4, 1, 3};

    multimodal::FloatMatrix expected;
    std::string error;
    CHECK(multimodal::embed_patches_at_positions(
        patches, positions, weights, &expected, &error));

    CudaExecutionContext context;
    multimodal::CudaPatchEmbedding embedding;
    CudaTensor output;
    CHECK(CudaExecutionContext::create(&context).is_ok());
    CHECK(multimodal::CudaPatchEmbedding::upload(
        &context, weights, &embedding).is_ok());
    CHECK(embedding.valid());
    CHECK(CudaTensor::create_contiguous(
        {3, 3}, DType::Float32, &output).is_ok());
    CHECK(embedding.forward(&context, patches, positions, &output).is_ok());

    std::vector<float> actual(expected.values.size());
    CHECK(context.download(
        output, actual.data(), actual.size() * sizeof(float)).is_ok());
    CHECK(context.synchronize().is_ok());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK(std::fabs(actual[index] - expected.values[index]) <= 1.0e-6F);
    }

    CHECK(!embedding.forward(
        &context, patches, {4, 1, 5}, &output).is_ok());
    CudaTensor wrong_output;
    CHECK(CudaTensor::create_contiguous(
        {3, 2}, DType::Float32, &wrong_output).is_ok());
    CHECK(!embedding.forward(
        &context, patches, positions, &wrong_output).is_ok());

    std::cout << "multimodal_cuda_vision_embedding_test: PASS\n";
    return 0;
}
