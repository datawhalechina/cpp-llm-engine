// Engine/Chapter10: compose text tokenization, CUDA execution, and HTTP/SSE.
#include "cpp_llm_engine/real_deepseek2_cuda_executor_factory.h"
#include "cpp_llm_engine/real_tokenizer_backend.h"

#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/cpp_httplib_transport.h"
#include "chapter10_serving/http_adapter.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: real_deepseek2_cuda_http_server <model.gguf> "
                     "[port] [bind-address]\n";
        return 2;
    }
    const int port = argc >= 3 ? std::atoi(argv[2]) : 18085;
    const std::string bind_address = argc >= 4 ? argv[3] : "127.0.0.1";
    if (port <= 0 || port > 65535) {
        std::cerr << "port must be in [1, 65535]\n";
        return 2;
    }

    std::string error;
    auto tokenizer = cpp_llm_engine::RealTokenizerBackend::load(argv[1], &error);
    auto factory = cpp_llm_engine::RealDeepSeek2CudaExecutorFactory::load(
        argv[1], 256, &error);
    if (tokenizer == nullptr || factory == nullptr) {
        std::cerr << "DeepSeek-V2-Lite CUDA HTTP load failed: " << error << '\n';
        return 1;
    }
    if (tokenizer->vocab_size() != factory->vocab_size()) {
        std::cerr << "tokenizer and CUDA model vocabulary sizes differ\n";
        return 1;
    }
    if (!factory->configure_paged_mla_cache(128, 8, &error)) {
        std::cerr << "CUDA paged MLA setup failed: " << error << '\n';
        return 1;
    }

    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = factory->context_capacity();
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter10_serving::ServiceCore service(
        16, 8, {}, factory->paged_mla_resource(), {}, factory.get());
    chapter10_serving::HttpAdapter adapter(
        service,
        runtime,
        sampler,
        [&factory]() { return factory->create_executor(); },
        [&tokenizer](const std::string& prompt) {
            return tokenizer->encode(prompt);
        });

    std::cout << "DeepSeek-V2-Lite CUDA HTTP server: http://" << bind_address
              << ':' << port << "\nmodel layers: " << factory->layer_count()
              << "\nrequest body: {\"request_id\":1,\"prompt\":\"Hello\"}"
              << std::endl;
    chapter10_serving::CppHttplibServer server(adapter, bind_address, port);
    return server.listen();
}
