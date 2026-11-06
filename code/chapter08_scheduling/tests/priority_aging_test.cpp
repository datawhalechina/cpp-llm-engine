// Chapter 08：验证基础优先级、有界老化和确定性平局规则。
// 职责：固定请求选择策略，不测试服务准入、Deadline 或抢占。
// 验证入口：chapter08_priority_aging_test。
#include "chapter08_scheduling/step_scheduler.h"

#include "chapter06_generation/request.h"
#include "chapter06_generation/sampler.h"
#include "cpp_llm_engine/testing/test.h"

#include <vector>

int main() {
    using namespace chapter06_generation;
    using namespace chapter08_scheduling;
    cpp_llm_engine::testing::TestSuite suite("chapter08_priority_aging");

    suite.run("higher base priority is selected first", [&] {
        Request low({1});
        Request high({2});
        GreedySampler sampler;
        const std::vector<ScheduledRequest> requests{
            {10, &low, &sampler, {}, {0, 0}},
            {11, &high, &sampler, {}, {2, 1}},
        };

        const StepBatchPlan plan = StepBatchPlanner{}.build(requests, 1, 1);
        CPP_LLM_REQUIRE(suite, plan.candidates.size() == 1);
        CPP_LLM_CHECK(suite, plan.candidates.front().request_id == 11);
    });

    suite.run("enqueue sequence breaks equal priority ties", [&] {
        Request later({1});
        Request earlier({2});
        GreedySampler sampler;
        const std::vector<ScheduledRequest> requests{
            {20, &later, &sampler, {}, {1, 9}},
            {21, &earlier, &sampler, {}, {1, 3}},
        };

        const StepBatchPlan plan = StepBatchPlanner{}.build(requests, 1, 1);
        CPP_LLM_REQUIRE(suite, plan.candidates.size() == 1);
        CPP_LLM_CHECK(suite, plan.candidates.front().request_id == 21);
    });

    suite.run("bounded aging eventually matches a small priority gap", [&] {
        Request low({1});
        GreedySampler sampler;
        PriorityAgingPolicy policy({1, 3});
        bool low_selected = false;

        for (std::size_t step = 0; step < 3; ++step) {
            Request fresh_high({2});
            const std::size_t high_id = 100 + step;
            const std::vector<ScheduledRequest> requests{
                {30, &low, &sampler, {}, {0, 0}},
                {high_id, &fresh_high, &sampler, {}, {2, step + 1}},
            };
            const std::size_t version_before = policy.version();
            const StepBatchPlan plan = StepBatchPlanner{}.build(
                requests, 1, 1, 0, &policy);
            CPP_LLM_CHECK(suite, policy.version() == version_before);
            CPP_LLM_REQUIRE(suite, plan.candidates.size() == 1);
            const std::size_t selected = plan.candidates.front().request_id;
            policy.finish_step(requests, {selected});
            if (selected == 30) {
                low_selected = true;
                CPP_LLM_CHECK(suite, step == 2);
                break;
            }
        }
        CPP_LLM_CHECK(suite, low_selected);
    });

    suite.run("aging boost stops at the configured limit", [&] {
        Request low({1});
        Request high({2});
        GreedySampler sampler;
        PriorityAgingPolicy policy({1, 3});
        const std::vector<ScheduledRequest> requests{
            {40, &low, &sampler, {}, {0, 0}},
            {41, &high, &sampler, {}, {10, 1}},
        };

        for (int step = 0; step < 10; ++step) {
            policy.finish_step(requests, {41});
        }
        CPP_LLM_CHECK(suite, policy.waiting_steps(40) == 10);
        CPP_LLM_CHECK(suite, policy.effective_priority(requests[0]) == 3);
        const StepBatchPlan plan = StepBatchPlanner{}.build(
            requests, 1, 1, 0, &policy);
        CPP_LLM_REQUIRE(suite, plan.candidates.size() == 1);
        CPP_LLM_CHECK(suite, plan.candidates.front().request_id == 41);
    });

    suite.run("zero aging interval is rejected", [&] {
        bool rejected = false;
        try {
            static_cast<void>(PriorityAgingPolicy({0, 1}));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
    });

    return suite.finish();
}
