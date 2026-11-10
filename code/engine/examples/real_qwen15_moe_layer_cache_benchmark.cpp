// Chapter 09：比较 Qwen1.5-MoE 逐层重载与全层 LRU 缓存。
// 运行：... <model.gguf> [cache_layers] [uncached-first|cached-first]。
#include "cpp_llm_engine/real_qwen2_moe_executor_factory.h"

#include "chapter06_generation/sampler.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Result {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int prefill_token = -1;
    int decode_token = -1;
    cpp_llm_engine::RealQwen2MoeLayerCacheStats cache;
};

bool run(cpp_llm_engine::RealQwen2MoeExecutorFactory& factory,
         std::size_t cache_layers,
         Result* result,
         std::string* error) {
    factory.set_layer_cache_capacity(cache_layers);
    auto executor = factory.create_executor();
    chapter06_generation::GreedySampler sampler;
    chapter06_generation::Logits prefill_logits;
    chapter06_generation::Logits decode_logits;
    const auto prefill_start = std::chrono::steady_clock::now();
    if (!executor->prefill({0}, &prefill_logits, error)) return false;
    const auto prefill_end = std::chrono::steady_clock::now();
    result->prefill_token = sampler.sample(prefill_logits);
    const auto decode_start = std::chrono::steady_clock::now();
    if (!executor->decode(result->prefill_token, &decode_logits, error)) {
        return false;
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result->decode_token = sampler.sample(decode_logits);
    result->prefill_ms = std::chrono::duration<double, std::milli>(
        prefill_end - prefill_start).count();
    result->decode_ms = std::chrono::duration<double, std::milli>(
        decode_end - decode_start).count();
    result->cache = factory.layer_cache_stats();
    return true;
}

void print(const char* name, const Result& result) {
    std::cout << name << "_prefill_ms: " << result.prefill_ms << '\n'
              << name << "_decode_ms: " << result.decode_ms << '\n'
              << name << "_cache_hits: " << result.cache.hits << '\n'
              << name << "_cache_misses: " << result.cache.misses << '\n'
              << name << "_resident_layers: "
              << result.cache.resident_layers << '\n'
              << name << "_resident_bytes: "
              << result.cache.resident_bytes << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: real_qwen15_moe_layer_cache_benchmark "
                     "<model.gguf> [cache_layers] "
                     "[uncached-first|cached-first]\n";
        return 2;
    }
    const std::size_t cache_layers = argc == 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 24;
    const std::string order = argc == 4 ? argv[3] : "uncached-first";
    std::string error;
    auto factory = cpp_llm_engine::RealQwen2MoeExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr || cache_layers > factory->layer_count() ||
        (order != "uncached-first" && order != "cached-first")) {
        std::cerr << "benchmark setup failed: " << error << '\n';
        return 1;
    }
    Result uncached;
    Result cached;
    const bool ran = order == "uncached-first"
        ? run(*factory, 0, &uncached, &error) &&
              run(*factory, cache_layers, &cached, &error)
        : run(*factory, cache_layers, &cached, &error) &&
              run(*factory, 0, &uncached, &error);
    if (!ran) {
        std::cerr << "benchmark execution failed: " << error << '\n';
        return 1;
    }
    std::cout << "path_order: " << order << '\n';
    print("uncached", uncached);
    print("cached", cached);
    const bool tokens_match =
        uncached.prefill_token == cached.prefill_token &&
        uncached.decode_token == cached.decode_token;
    std::cout << "tokens_match: " << (tokens_match ? "true" : "false")
              << "\nqwen15_moe_layer_cache_benchmark: "
              << (tokens_match ? "PASS" : "FAIL") << '\n';
    return tokens_match ? 0 : 1;
}
