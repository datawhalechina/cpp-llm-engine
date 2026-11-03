// Chapter 09：启动可选 cpp-httplib HTTP transport。
// 职责：组装教学 Runtime、ServiceCore 和真实 HTTP server，不复制服务状态。
// 验证入口：开启 CHAPTER09_ENABLE_CPPHTTPLIB 后运行本示例并发送 loopback 请求。
#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "chapter09_serving/cpp_httplib_transport.h"

#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    using namespace chapter06_generation;
    using namespace chapter09_serving;

    const int port = argc > 1 ? std::atoi(argv[1]) : 18080;
    GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 32;
    Embedding embedding(32);
    PositionEncoding positions(16);
    Transformer transformer(2, embedding, positions);
    Runtime runtime(config, transformer);
    GreedySampler sampler;
    ServingCore service(16, 4);
    HttpAdapter adapter(service, runtime, sampler);
    CppHttplibServer server(adapter, "127.0.0.1", port);
    return server.listen();
}
