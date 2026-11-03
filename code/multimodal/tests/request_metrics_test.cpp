// 多模态 Runtime 测试：验证请求阶段耗时、重复视觉调用和峰值存活字节。
// 覆盖：注入时钟、阶段匹配、shape、聚合、非法枚举和时钟回退。
// 依赖：不需要真实模型、Python、网络、CUDA 或 BLAS。
// 运行：由 CTest 执行 multimodal_request_metrics_test。
#include "multimodal/request_metrics.h"
#include "cpp_llm_engine/testing/test.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

bool near(double lhs, double rhs) {
    return std::abs(lhs - rhs) < 1.0e-12;
}

class ManualClock final {
public:
    multimodal::MultimodalMetricsClock::time_point now() const { return now_; }
    void advance(std::chrono::milliseconds duration) { now_ += duration; }
    void rewind(std::chrono::milliseconds duration) { now_ -= duration; }

private:
    multimodal::MultimodalMetricsClock::time_point now_{};
};

}  // namespace

int main() {
    using multimodal::MultimodalRequestStage;

    cpp_llm_engine::testing::TestSuite suite("multimodal_request_metrics");
    suite.run("阶段计时、形状和峰值存活字节可重复聚合", [&] {
    ManualClock clock;
    multimodal::MultimodalStageRecorder recorder([&clock]() { return clock.now(); });
    std::string error;
    CPP_LLM_REQUIRE(suite, recorder.begin(MultimodalRequestStage::ImageDecode, 100, &error));
    CPP_LLM_CHECK(suite, recorder.active());
    CPP_LLM_CHECK(suite, !recorder.begin(MultimodalRequestStage::DynamicTiling, 0, &error));
    CPP_LLM_CHECK(suite, error.find("already active") != std::string::npos);
    CPP_LLM_CHECK(suite, !recorder.finish(MultimodalRequestStage::DynamicTiling, 0, &error));
    CPP_LLM_CHECK(suite, error.find("does not match") != std::string::npos);
    clock.advance(std::chrono::milliseconds(10));
    CPP_LLM_REQUIRE(suite, recorder.finish(MultimodalRequestStage::ImageDecode, 120, &error));
    CPP_LLM_CHECK(suite, !recorder.active());

    for (int visual = 0; visual < 2; ++visual) {
        CPP_LLM_REQUIRE(suite, recorder.begin(
            MultimodalRequestStage::VisionEncoder, 200 + visual * 50, &error));
        clock.advance(std::chrono::milliseconds(20));
        CPP_LLM_REQUIRE(suite, recorder.finish(
            MultimodalRequestStage::VisionEncoder, 300 + visual * 50, &error));
        CPP_LLM_REQUIRE(suite, recorder.begin(MultimodalRequestStage::Projector, 180, &error));
        clock.advance(std::chrono::milliseconds(5));
        CPP_LLM_REQUIRE(suite, recorder.finish(MultimodalRequestStage::Projector, 190, &error));
    }
    CPP_LLM_REQUIRE(suite, recorder.set_sequence_shape(2, 128, 166, &error));

    const auto& metrics = recorder.metrics();
    CPP_LLM_CHECK(suite, metrics.stage(MultimodalRequestStage::ImageDecode).invocation_count == 1);
    CPP_LLM_CHECK(suite, metrics.stage(MultimodalRequestStage::VisionEncoder).invocation_count == 2);
    CPP_LLM_CHECK(suite, near(metrics.stage(MultimodalRequestStage::VisionEncoder).elapsed_seconds, 0.04));
    CPP_LLM_CHECK(suite, metrics.stage(MultimodalRequestStage::Projector).invocation_count == 2);
    CPP_LLM_CHECK(suite, near(metrics.stage(MultimodalRequestStage::Projector).elapsed_seconds, 0.01));
    CPP_LLM_CHECK(suite, metrics.visual_input_count == 2);
    CPP_LLM_CHECK(suite, metrics.visual_row_count == 128);
    CPP_LLM_CHECK(suite, metrics.prompt_position_count == 166);
    CPP_LLM_CHECK(suite, metrics.peak_live_bytes == 350);
    CPP_LLM_CHECK(suite, near(metrics.input_pipeline_seconds(), 0.06));
    CPP_LLM_CHECK(suite, std::string(multimodal::multimodal_request_stage_name(
        MultimodalRequestStage::EmbeddingComposition)) == "embedding_composition");

    CPP_LLM_CHECK_THROWS(
        suite, std::out_of_range,
        (void)metrics.stage(MultimodalRequestStage::Count));
    CPP_LLM_CHECK(suite, !recorder.set_sequence_shape(2, 1, 3, &error));
    CPP_LLM_CHECK(suite, !recorder.finish(MultimodalRequestStage::ImageDecode, 0, &error));

    CPP_LLM_REQUIRE(suite, recorder.begin(MultimodalRequestStage::LanguagePrefill, 400, &error));
    clock.rewind(std::chrono::milliseconds(1));
    CPP_LLM_CHECK(suite, !recorder.finish(MultimodalRequestStage::LanguagePrefill, 400, &error));
    CPP_LLM_CHECK(suite, recorder.active());
    });
    return suite.finish();
}
