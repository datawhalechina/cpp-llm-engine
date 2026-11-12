// Engine CUDA: validate all DeepSeek2 layers, final norm and LM Head.
#include "cpp_llm_engine/model/real_deepseek2_cpu_model.h"
#include "cpp_llm_engine/model/real_deepseek2_cuda_model.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
    std::cerr << "check failed line " << __LINE__ << '\n'; return 1; \
} } while (false)

namespace {

std::size_t argmax(const std::vector<float>& values) {
    return static_cast<std::size_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
}

double cosine(const std::vector<float>& left,
              const std::vector<float>& right) {
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        dot += static_cast<double>(left[index]) * right[index];
        left_norm += static_cast<double>(left[index]) * left[index];
        right_norm += static_cast<double>(right[index]) * right[index];
    }
    return dot / std::sqrt(left_norm * right_norm);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: real_deepseek2_cuda_model_test <model.gguf>\n";
        return 2;
    }
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    std::string error;
    auto cpu_model = cpp_llm_engine::model::RealDeepSeek2CpuModel::load(
        argv[1], &error);
    CHECK(cpu_model != nullptr);
    auto cuda_model = cpp_llm_engine::model::RealDeepSeek2CudaModel::load(
        argv[1], 0, &error);
    CHECK(cuda_model != nullptr);
    CHECK(cuda_model->layer_count() == 27);
    auto cpu = cpu_model->create_context();
    auto gpu = cuda_model->create_context(2, &error);
    CHECK(cpu != nullptr && gpu != nullptr);

    constexpr std::int32_t token = 11;
    std::vector<float> cpu_logits;
    std::vector<float> cuda_logits;
    CHECK(cpu->decode_token(token, &cpu_logits, &error));
    const auto status = gpu->forward_token(token, &cuda_logits);
    if (!status.is_ok()) std::cerr << status.message() << '\n';
    CHECK(status.is_ok());
    CHECK(cpu_logits.size() == cuda_logits.size());
    const double similarity = cosine(cpu_logits, cuda_logits);
    const std::size_t cpu_token = argmax(cpu_logits);
    const std::size_t cuda_token = argmax(cuda_logits);
    std::cout << "deepseek_cuda_model_layers=" << cuda_model->layer_count() << '\n'
              << "deepseek_cuda_model_logit_cosine=" << similarity << '\n'
              << "deepseek_cuda_model_cpu_token=" << cpu_token << '\n'
              << "deepseek_cuda_model_cuda_token=" << cuda_token << '\n'
              << "deepseek_cuda_model_position=" << gpu->position() << '\n';
    CHECK(similarity >= 0.999);
    CHECK(cpu_token == cuda_token);
    CHECK(gpu->position() == 1);
    CHECK(gpu->layer_count() == 27);
    return 0;
}

#undef CHECK
