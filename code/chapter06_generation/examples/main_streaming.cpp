// Chapter 06：运行 Serving Preview 的 streaming callback 演示。
// 输出：Scheduler 交错推进请求时的 per-request token emission。
// 验证：使用 scripts/run.sh 构建，并通过 CTest 检查 streaming output。
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/scheduler.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"

#include <iostream>
#include <vector>

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

    Request request_a({1, 2, 3});
    Request request_b({4, 5});

    std::vector<ScheduledRequest> requests;
    requests.push_back({0, &request_a, &sampler, {}});
    requests.push_back({1, &request_b, &sampler, {}});

    const RoundRobinScheduler scheduler;
    const SchedulerStats stats = scheduler.run_with_stats(
        runtime,
        requests,
        [](std::size_t request_id, TokenId token, std::size_t step) {
            std::cout << "[chapter06] stream request=" << request_id
                      << " step=" << step
                      << " token=" << token << "\n";
        });

    std::cout << "[chapter06] streaming runtime skeleton\n";
    std::cout << "[chapter06] ready: " << std::boolalpha << runtime.ready() << "\n";
    std::cout << "[chapter06] scheduled requests: " << stats.scheduled_requests << "\n";
    std::cout << "[chapter06] prefetched requests: " << stats.prefetched_requests << "\n";
    std::cout << "[chapter06] generated steps: " << stats.generated_tokens << "\n";
    std::cout << "[chapter06] completed requests: " << stats.completed_requests << "\n";
    std::cout << "[chapter06] request_a cache size: " << request_a.cache().size() << "\n";
    std::cout << "[chapter06] request_b cache size: " << request_b.cache().size() << "\n";
    std::cout << "[chapter06] request_a phase: " << request_phase_name(request_a.phase()) << "\n";
    std::cout << "[chapter06] request_b phase: " << request_phase_name(request_b.phase()) << "\n";
    std::cout << "[chapter06] request_a finished: " << std::boolalpha << request_a.finished() << "\n";
    std::cout << "[chapter06] request_b finished: " << std::boolalpha << request_b.finished() << "\n";
    return 0;
}
