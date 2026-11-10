// Engine/Chapter10：把 DeepSeek-V2-Lite Factory 接入 cpp-httplib Transport。
#include "cpp_llm_engine/real_deepseek2_executor_factory.h"

#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/cpp_httplib_transport.h"
#include "chapter10_serving/http_adapter.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: real_deepseek2_http_server <model.gguf> [port] "
                     "[bind-address]\n";
        return 2;
    }
    const int port = argc >= 3 ? std::atoi(argv[2]) : 18084;
    const std::string bind_address = argc >= 4 ? argv[3] : "127.0.0.1";
    if (port <= 0 || port > 65535) {
        std::cerr << "port must be in [1, 65535]\n";
        return 2;
    }

    std::string error;
    auto factory = cpp_llm_engine::RealDeepSeek2ExecutorFactory::load(
        argv[1], &error);
    if (factory == nullptr) {
        std::cerr << "DeepSeek-V2-Lite HTTP server load failed: "
                  << error << '\n';
        return 1;
    }
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 2048;
    config.eos_token_id = factory->eos_token_id();
    chapter06_generation::Runtime runtime(config);
    chapter06_generation::GreedySampler sampler;
    chapter10_serving::ServiceCore service(16, 16, {}, {}, {}, factory.get());
    chapter10_serving::HttpAdapter adapter(
        service,
        runtime,
        sampler,
        [&factory]() { return factory->create_executor(); });

    std::cout << "DeepSeek-V2-Lite HTTP server: http://" << bind_address
              << ':' << port << "\nmodel layers: " << factory->layer_count()
              << "\nexample prompt_tokens: 0" << std::endl;
    chapter10_serving::CppHttplibServer server(adapter, bind_address, port);
    return server.listen();
}
