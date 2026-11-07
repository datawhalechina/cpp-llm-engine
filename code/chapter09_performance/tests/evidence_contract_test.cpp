// Chapter 09：验证统一证据记录的正确性闸门、字段约束和 JSON 结构。
// 职责：防止 Benchmark 与 Profiling 再次产生互不兼容的身份字段。
// 验证入口：chapter09_evidence_contract_test。
#include "chapter09_performance/evidence_record.h"

#include "cpp_llm_engine/testing/test.h"
#include "support/sample_records.h"

#include <string>

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter09_evidence_contract");

    suite.run("complete record is valid", [&] {
        const auto record = chapter09_performance::testing::sample_record("EXP-BASELINE");
        std::string error;
        CPP_LLM_CHECK(suite, record.valid(&error));
        CPP_LLM_CHECK(suite, error.empty());
        CPP_LLM_CHECK(suite, record.summary().total_ms.median == 10.0);
    });

    suite.run("correctness failure blocks evidence", [&] {
        auto record = chapter09_performance::testing::sample_record("EXP-WRONG");
        record.correctness_pass = false;
        std::string error;
        CPP_LLM_CHECK(suite, !record.valid(&error));
        CPP_LLM_CHECK(suite, error.find("correctness") != std::string::npos);
    });

    suite.run("sample count must match workload", [&] {
        auto record = chapter09_performance::testing::sample_record("EXP-SHORT");
        record.samples.pop_back();
        std::string error;
        CPP_LLM_CHECK(suite, !record.valid(&error));
        CPP_LLM_CHECK(suite, error.find("sample count") != std::string::npos);
    });

    suite.run("ragged prompts describe one length per request", [&] {
        auto record = chapter09_performance::testing::sample_record("EXP-RAGGED");
        record.workload.prompt_tokens_per_request = 0;
        record.workload.prompt_tokens = {6, 1};
        record.workload.request_count = 2;
        std::string error;
        CPP_LLM_CHECK(suite, record.valid(&error));

        record.workload.prompt_tokens = {6};
        CPP_LLM_CHECK(suite, !record.valid(&error));
        CPP_LLM_CHECK(suite, error.find("workload") != std::string::npos);
    });

    suite.run("json keeps identity and raw samples", [&] {
        const auto record = chapter09_performance::testing::sample_record("EXP-JSON");
        const std::string json = chapter09_performance::render_evidence_json(record);
        CPP_LLM_CHECK(suite, json.find("\"schema_version\": 1") != std::string::npos);
        CPP_LLM_CHECK(suite, json.find("\"workload_id\": \"WL-CPU-BATCH-V1\"") !=
                                 std::string::npos);
        CPP_LLM_CHECK(suite, json.find("\"samples\": [") != std::string::npos);
        CPP_LLM_CHECK(suite, json.find("\"correctness_checks\": [") !=
                                 std::string::npos);
        CPP_LLM_CHECK(suite, json.find("\"total_ms_median\": 10.0000") !=
                                 std::string::npos);
    });

    suite.run("json escapes identity control characters", [&] {
        auto record = chapter09_performance::testing::sample_record("EXP-ESCAPE");
        record.experiment_id = std::string("EXP\n\"\\") + static_cast<char>(1);
        const std::string json = chapter09_performance::render_evidence_json(record);
        CPP_LLM_CHECK(suite, json.find("EXP\\n\\\"\\\\\\u0001") != std::string::npos);
    });

    return suite.finish();
}
