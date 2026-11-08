// Chapter 09：验证性能对比的可比性和保护指标策略。
// 职责：锁定四类决策，避免错误或不同 workload 进入性能结论。
// 验证入口：chapter09_comparison_test。
#include "chapter09_performance/comparison.h"

#include "cpp_llm_engine/testing/test.h"
#include "support/sample_records.h"

#include <cmath>
#include <limits>

int main() {
    using chapter09_performance::Decision;
    cpp_llm_engine::testing::TestSuite suite("chapter09_comparison");

    suite.run("faster comparable experiment is kept", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        auto experiment = chapter09_performance::testing::sample_record("EXP-B");
        for (auto& sample : experiment.samples) {
            sample.total_ms *= 0.9;
            sample.tokens_per_second *= 1.1;
        }
        const auto result = chapter09_performance::compare(baseline, experiment);
        CPP_LLM_CHECK(suite, result.decision == Decision::keep);
        CPP_LLM_CHECK(suite, result.latency_delta_percent < 0.0);
        CPP_LLM_CHECK(suite, result.throughput_delta_percent > 0.0);
    });

    suite.run("latency regression is rejected", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        auto experiment = chapter09_performance::testing::sample_record("EXP-SLOW");
        for (auto& sample : experiment.samples) sample.total_ms *= 1.2;
        const auto result = chapter09_performance::compare(baseline, experiment);
        CPP_LLM_CHECK(suite, result.decision == Decision::reject);
    });

    suite.run("unchanged primary metric is rejected", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        const auto experiment = chapter09_performance::testing::sample_record("EXP-SAME");
        const auto result = chapter09_performance::compare(baseline, experiment);
        CPP_LLM_CHECK(suite, result.decision == Decision::reject);
        CPP_LLM_CHECK(suite, result.reason.find("primary metric") != std::string::npos);
    });

    suite.run("latency can be selected as primary metric", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        auto experiment = chapter09_performance::testing::sample_record("EXP-LOWER-LATENCY");
        for (auto& sample : experiment.samples) sample.total_ms *= 0.9;
        chapter09_performance::ComparisonPolicy policy;
        policy.primary_metric = chapter09_performance::PrimaryMetric::latency;
        CPP_LLM_CHECK(suite,
                      chapter09_performance::compare(baseline, experiment, policy).decision ==
                          Decision::keep);
    });

    suite.run("different workload is not comparable", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        auto experiment = chapter09_performance::testing::sample_record("EXP-OTHER");
        experiment.workload_id = "WL-CPU-BATCH-V2";
        const auto result = chapter09_performance::compare(baseline, experiment);
        CPP_LLM_CHECK(suite, result.decision == Decision::not_comparable);
        CPP_LLM_CHECK(suite, result.latency_delta_percent == 0.0);
    });

    suite.run("different hardware is not comparable", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        auto experiment = chapter09_performance::testing::sample_record("EXP-HARDWARE");
        experiment.environment.hardware = "another CPU";
        CPP_LLM_CHECK(suite,
                      chapter09_performance::compare(baseline, experiment).decision ==
                          Decision::not_comparable);

        experiment.environment = baseline.environment;
        experiment.environment.hardware_concurrency = 16;
        CPP_LLM_CHECK(suite,
                      chapter09_performance::compare(baseline, experiment).decision ==
                          Decision::not_comparable);
    });

    suite.run("different ragged prompts are not comparable", [&] {
        auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        auto experiment = chapter09_performance::testing::sample_record("EXP-B");
        baseline.workload.prompt_tokens_per_request = 0;
        experiment.workload.prompt_tokens_per_request = 0;
        baseline.workload.prompt_tokens = {6, 1};
        experiment.workload.prompt_tokens = {6, 2};
        baseline.workload.request_count = 2;
        experiment.workload.request_count = 2;
        CPP_LLM_CHECK(suite,
                      chapter09_performance::compare(baseline, experiment).decision ==
                          Decision::not_comparable);
    });

    suite.run("different measurement protocol is not comparable", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        auto different_warmup = chapter09_performance::testing::sample_record("EXP-WARMUP");
        different_warmup.workload.warmup_iterations = 0;
        CPP_LLM_CHECK(suite,
                      chapter09_performance::compare(baseline, different_warmup).decision ==
                          Decision::not_comparable);

        auto different_samples = chapter09_performance::testing::sample_record("EXP-SAMPLES");
        different_samples.workload.measurement_iterations = 2;
        different_samples.samples.pop_back();
        CPP_LLM_CHECK(suite,
                      chapter09_performance::compare(baseline, different_samples).decision ==
                          Decision::not_comparable);
    });

    suite.run("failed correctness is invalid", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        auto experiment = chapter09_performance::testing::sample_record("EXP-WRONG");
        experiment.correctness_pass = false;
        const auto result = chapter09_performance::compare(baseline, experiment);
        CPP_LLM_CHECK(suite, result.decision == Decision::invalid);
    });

    suite.run("memory growth from a zero baseline is rejected", [&] {
        auto baseline = chapter09_performance::testing::sample_record("EXP-ZERO-MEMORY");
        auto experiment = chapter09_performance::testing::sample_record("EXP-MEMORY");
        for (auto& sample : baseline.samples) sample.peak_memory_mb = 0.0;
        for (auto& sample : experiment.samples) {
            sample.total_ms *= 0.9;
            sample.tokens_per_second *= 1.1;
        }
        const auto result = chapter09_performance::compare(baseline, experiment);
        CPP_LLM_CHECK(suite, result.decision == Decision::reject);
        CPP_LLM_CHECK(suite, std::isinf(result.memory_delta_percent));
    });

    suite.run("invalid comparison policy is rejected", [&] {
        const auto baseline = chapter09_performance::testing::sample_record("EXP-A");
        const auto experiment = chapter09_performance::testing::sample_record("EXP-B");
        chapter09_performance::ComparisonPolicy policy;
        policy.max_latency_regression_percent = -1.0;
        CPP_LLM_CHECK(
            suite,
            chapter09_performance::compare(baseline, experiment, policy).decision ==
                Decision::invalid);
        policy.max_latency_regression_percent =
            std::numeric_limits<double>::quiet_NaN();
        CPP_LLM_CHECK(
            suite,
            chapter09_performance::compare(baseline, experiment, policy).decision ==
                Decision::invalid);
    });

    return suite.finish();
}
