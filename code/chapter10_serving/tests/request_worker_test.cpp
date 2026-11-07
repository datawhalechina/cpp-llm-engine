// Chapter 10：请求级 CPU Worker 并行隔离测试。
// 职责：证明独立 Executor 的 prefill 可以重叠，且生成结果不会跨请求混合。
// 验证入口：chapter10_request_worker_test 或带 worker 标签的 CTest。
#include "chapter06_generation/request.h"
#include "chapter06_generation/runtime.h"
#include "chapter06_generation/sampler.h"
#include "chapter10_serving/request_worker.h"
#include "chapter10_serving/service_core.h"
#include "cpp_llm_engine/testing/test.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace {

struct ParallelGate {
    std::mutex mutex;
    std::condition_variable ready;
    std::size_t arrived = 0;
    std::size_t peak = 0;
};

class CoordinatedExecutor final : public chapter06_generation::RequestModelExecutor {
public:
    CoordinatedExecutor(
        std::shared_ptr<ParallelGate> gate,
        chapter06_generation::TokenId token)
        : gate_(std::move(gate)), token_(token) {}

    bool prefill(
        const chapter06_generation::TokenSequence&,
        chapter06_generation::Logits* logits,
        std::string* error) override {
        std::unique_lock<std::mutex> lock(gate_->mutex);
        ++gate_->arrived;
        gate_->peak = std::max(gate_->peak, gate_->arrived);
        gate_->ready.notify_all();
        if (!gate_->ready.wait_for(
                lock, std::chrono::seconds(2), [this] { return gate_->arrived >= 2; })) {
            if (error != nullptr) *error = "request workers did not overlap";
            return false;
        }
        lock.unlock();
        logits->assign(8, 0.0f);
        (*logits)[static_cast<std::size_t>(token_)] = 1.0f;
        return true;
    }

    bool decode(
        chapter06_generation::TokenId,
        chapter06_generation::Logits*,
        std::string*) override {
        return false;
    }

private:
    std::shared_ptr<ParallelGate> gate_;
    chapter06_generation::TokenId token_;
};

}  // namespace

int main() {
    using namespace chapter06_generation;
    cpp_llm_engine::testing::TestSuite suite("chapter10_request_worker");

    suite.run("two request workers overlap and preserve token ownership", [&] {
        GenerationConfig config;
        config.max_new_tokens = 1;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request_a({1});
        Request request_b({2});
        auto gate = std::make_shared<ParallelGate>();
        chapter10_serving::ServiceCore service(2, 2);

        CPP_LLM_REQUIRE(suite, service.admit(
            35, request_a, sampler, std::make_shared<CoordinatedExecutor>(gate, 3)).accepted());
        CPP_LLM_REQUIRE(suite, service.admit(
            36, request_b, sampler, std::make_shared<CoordinatedExecutor>(gate, 4)).accepted());
        const auto stats = service.dispatch_parallel_requests(runtime, 2);

        CPP_LLM_CHECK_EQ(suite, stats.scheduled_requests, 2U);
        CPP_LLM_CHECK_EQ(suite, stats.completed, 2U);
        CPP_LLM_CHECK_EQ(suite, stats.generated_tokens, 2U);
        CPP_LLM_CHECK_EQ(suite, gate->peak, 2U);
        CPP_LLM_CHECK(suite, request_a.generated() == TokenSequence({3}));
        CPP_LLM_CHECK(suite, request_b.generated() == TokenSequence({4}));
        CPP_LLM_CHECK_EQ(suite, service.release_terminal(), 2U);
    });

    suite.run("workers reject shared mutable request state", [&] {
        GenerationConfig config;
        Runtime runtime(config);
        GreedySampler sampler;
        Request request({1});
        auto gate = std::make_shared<ParallelGate>();
        auto executor = std::make_shared<CoordinatedExecutor>(gate, 3);
        const chapter10_serving::RequestWorkerPool workers;
        CPP_LLM_CHECK_THROWS(
            suite,
            std::invalid_argument,
            workers.run(
                runtime,
                {{40, &request, &sampler, executor},
                 {41, &request, &sampler,
                  std::make_shared<CoordinatedExecutor>(gate, 4)}},
                2));

        Request second({2});
        CPP_LLM_CHECK_THROWS(
            suite,
            std::invalid_argument,
            workers.run(
                runtime,
                {{42, &request, &sampler, executor},
                 {43, &second, &sampler, executor}},
                2));
    });

    return suite.finish();
}
