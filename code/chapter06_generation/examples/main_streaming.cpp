// Chapter 06：运行单请求逐词元回调演示。
// 输出：Runtime 提交词元后的 request-local step 和 token。
// 验证：使用 run_generation_core.sh 构建，并由 Generation 测试检查回调时序。
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"

#include <iostream>

int main() {
    using namespace chapter06_generation;

    GenerationConfig config;
    config.max_new_tokens = 2;
    config.use_kv_cache = true;

    Embedding embedding(32);
    PositionEncoding position_encoding(16);
    Transformer transformer(2, embedding, position_encoding);
    Runtime runtime(config, transformer);
    GreedySampler sampler;

    Request request({1, 2, 3});
    const std::size_t generated = runtime.generate(
        request,
        sampler,
        [](TokenId token, std::size_t step) {
            std::cout << "[chapter06] stream step=" << step
                      << " token=" << token << "\n";
        });

    std::cout << "[chapter06] single-request streaming callback\n";
    std::cout << "[chapter06] ready: " << std::boolalpha << runtime.ready() << "\n";
    std::cout << "[chapter06] generated steps: " << generated << "\n";
    std::cout << "[chapter06] token history size: " << request.history().size() << "\n";
    std::cout << "[chapter06] request phase: " << request_phase_name(request.phase()) << "\n";
    return 0;
}
