// Chapter 10: verify bounded request collection and per-request result delivery.
#include "chapter10_serving/http_adapter.h"

#include "cpp_llm_engine/testing/test.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>

int main() {
    using namespace chapter10_serving;
    cpp_llm_engine::testing::TestSuite suite("chapter10_request_preparer");

    suite.run("two concurrent preparations share one batch callback", [&] {
        CompletionRequestCollector collector(
            [](const std::vector<HttpRequest>& requests,
               std::vector<CompletionPreparationResult>* results) {
                results->clear();
                for (const HttpRequest& request : requests) {
                    results->push_back({
                        CompletionPrepareStatus::ready,
                        {{static_cast<int>(request.body.size())}, {}},
                        {},
                    });
                }
            },
            2,
            std::chrono::milliseconds(50));

        std::mutex start_mutex;
        std::condition_variable start_changed;
        std::size_t ready = 0;
        bool start = false;
        CompletionPreparationResult first;
        CompletionPreparationResult second;
        const auto run = [&](const char* body, CompletionPreparationResult* result) {
            {
                std::unique_lock<std::mutex> lock(start_mutex);
                ++ready;
                start_changed.notify_all();
                start_changed.wait(lock, [&] { return start; });
            }
            *result = collector.prepare({"POST", "/v1/completions", body});
        };
        std::thread first_thread(run, "one", &first);
        std::thread second_thread(run, "second", &second);
        {
            std::unique_lock<std::mutex> lock(start_mutex);
            start_changed.wait(lock, [&] { return ready == 2; });
            start = true;
        }
        start_changed.notify_all();
        first_thread.join();
        second_thread.join();

        CPP_LLM_CHECK(suite, first.status == CompletionPrepareStatus::ready);
        CPP_LLM_CHECK(suite, second.status == CompletionPrepareStatus::ready);
        CPP_LLM_CHECK_EQ(suite, first.prepared.prompt_positions.front(), 3);
        CPP_LLM_CHECK_EQ(suite, second.prepared.prompt_positions.front(), 6);
        const CompletionCollectorStats stats = collector.stats();
        CPP_LLM_CHECK_EQ(suite, stats.batch_calls, 1U);
        CPP_LLM_CHECK_EQ(suite, stats.prepared_requests, 2U);
        CPP_LLM_CHECK_EQ(suite, stats.max_batch_size, 2U);
    });

    suite.run("wrong callback cardinality fails every collected request", [&] {
        CompletionRequestCollector collector(
            [](const std::vector<HttpRequest>&,
               std::vector<CompletionPreparationResult>* results) {
                results->clear();
            },
            1,
            std::chrono::microseconds(0));
        const CompletionPreparationResult result = collector.prepare(
            {"POST", "/v1/completions", "one"});
        CPP_LLM_CHECK(
            suite,
            result.status == CompletionPrepareStatus::executor_unavailable);
        CPP_LLM_CHECK(
            suite,
            result.error.find("wrong result count") != std::string::npos);
    });

    return suite.finish();
}
