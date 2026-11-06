// Engine/Chapter 10：启动真实 TinyLlama CPU HTTP 服务。
// 职责：组装 tokenizer、共享 Model、每请求 Context、ServiceCore 和 cpp-httplib transport。
// 运行：real_tinyllama_http_server <model.gguf> [port] [bind-address]。
#include "cpp_llm_engine/real_llama_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/cpp_httplib_transport.h"
#include "chapter10_serving/http_adapter.h"
#include "chapter10_serving/service_core.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: real_tinyllama_http_server <model.gguf> [port] [bind-address]\n";
        return 2;
    }

    const int port = argc >= 3 ? std::atoi(argv[2]) : 18082;
    const std::string bind_address = argc >= 4 ? argv[3] : "127.0.0.1";
    if (port <= 0 || port > 65535) {
        std::cerr << "port must be in [1, 65535]\n";
        return 2;
    }

    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto factory = cpp_llm_engine::RealLlamaExecutorFactory::load(argv[1], &error);
    if (tokenizer == nullptr || factory == nullptr) {
        std::cerr << "real HTTP server load failed: " << error << '\n';
        return 1;
    }
    if (tokenizer->vocab_size() != factory->vocab_size()) {
        std::cerr << "tokenizer and model vocabulary sizes differ\n";
        return 1;
    }

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 4;
    config.max_context_tokens = 2048;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    if (!factory->configure_paged_kv_cache(128, 16, &error)) {
        std::cerr << "paged KV setup failed: " << error << '\n';
        return 1;
    }
    chapter10_serving::ServiceCore service(
        16, 16, {}, factory->paged_block_pool(), {}, factory.get());
    chapter10_serving::HttpAdapter adapter(
        service,
        runtime,
        sampler,
        [&factory]() { return factory->create_executor(); });

    const auto example_tokens = tokenizer->encode("Hello TinyLlama");
    std::cout << "real TinyLlama HTTP server: http://" << bind_address << ':' << port << '\n'
              << "model layers: " << factory->layer_count() << '\n'
              << "example prompt_tokens:";
    for (const auto token : example_tokens) std::cout << ' ' << token;
    std::cout << std::endl;

    chapter10_serving::CppHttplibServer server(adapter, bind_address, port);
    return server.listen();
}
