// Chapter 10：组装教学 Runtime、ServiceCore 和真实 cpp-httplib server。
// 职责：提供可选 socket 入口，不复制 Adapter 或服务生命周期状态。
// 验证入口：run_cpp_httplib_loopback.sh。
#include "chapter10_serving/cpp_httplib_transport.h"

#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"

#include <cstdlib>

int main(int argc, char** argv) {
    using namespace chapter06_generation;
    using namespace chapter10_serving;

    const int port = argc > 1 ? std::atoi(argv[1]) : 18081;
    GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 32;
    Transformer transformer(2, Embedding(32), PositionEncoding(16));
    Runtime runtime(config, transformer);
    GreedySampler sampler;
    ServiceCore service(16, 4);
    HttpAdapter adapter(service, runtime, sampler);
    CppHttplibServer server(adapter, "127.0.0.1", port);
    return server.listen();
}
