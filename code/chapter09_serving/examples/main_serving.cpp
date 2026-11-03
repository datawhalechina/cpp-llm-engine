// Chapter 09：运行最小 C++ Service Core 演示。
// 职责：提交 3 个请求，按 batch 上限分两轮执行，并展示完成后的 cache 回收。
// 验证：输出 admission、dispatch、completion 和 release 结果。
#include "chapter06_generation/embedding.h"
#include "chapter06_generation/position_encoding.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter06_generation/transformer.h"
#include "chapter09_serving/serving_core.h"

#include <iostream>

int main() {
    chapter06_generation::GenerationConfig config;
    config.max_new_tokens = 2;
    config.max_context_tokens = 32;
    chapter06_generation::Embedding embedding(32);
    chapter06_generation::PositionEncoding positions(16);
    chapter06_generation::Transformer transformer(2, embedding, positions);
    chapter06_generation::Runtime runtime(config, transformer);
    chapter06_generation::GreedySampler sampler;

    chapter06_generation::Request request_a({1, 2});
    chapter06_generation::Request request_b({3, 4});
    chapter06_generation::Request request_c({5, 6});
    chapter09_serving::ServingCore service(3, 2);

    std::cout << "admit_a=" << service.admit(1, request_a, sampler).accepted() << '\n'
              << "admit_b=" << service.admit(2, request_b, sampler).accepted() << '\n'
              << "admit_c=" << service.admit(3, request_c, sampler).accepted() << '\n';
    const auto first = service.dispatch(runtime);
    std::cout << "first_completed=" << first.completed << '\n'
              << "waiting_after_first=" << service.waiting_size() << '\n'
              << "released_first=" << service.release_completed() << '\n';
    const auto second = service.dispatch(runtime);
    std::cout << "second_completed=" << second.completed << '\n'
              << "released_second=" << service.release_completed() << '\n'
              << "inflight=" << service.inflight_size() << '\n'
              << "accepted_total=" << service.metrics().accepted_total << '\n'
              << "completed_total=" << service.metrics().completed_total << '\n'
              << "released_total=" << service.metrics().released_total << '\n'
              << "peak_inflight=" << service.metrics().peak_inflight << '\n';
    return request_a.finished() && request_b.finished() && request_c.finished() &&
                   request_a.cache().empty() && request_b.cache().empty() && request_c.cache().empty()
               ? 0
               : 1;
}
