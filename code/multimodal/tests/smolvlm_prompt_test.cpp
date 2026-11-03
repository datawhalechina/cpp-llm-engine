// 多模态 CPU 测试：验证 SmolVLM 模板和 IDEFICS3 chunk 展开顺序。
// 覆盖：overview-only、2x3 tiles、行分隔、visual index、空问题和非法网格。
// 运行：ctest --test-dir <multimodal-build> --output-on-failure。
#include "multimodal/smolvlm_prompt.h"
#include "cpp_llm_engine/testing/test.h"

#include <limits>
#include <string>

namespace {

using multimodal::PromptChunkKind;
using multimodal::SmolvlmPromptChunks;

void test_overview_only(cpp_llm_engine::testing::TestSuite& suite) {
    SmolvlmPromptChunks prompt;
    std::string error;
    CPP_LLM_REQUIRE(suite, multimodal::build_smolvlm_single_turn_prompt(
        "Describe it.", 0, 0, &prompt, &error));
    CPP_LLM_CHECK(suite, error.empty());
    CPP_LLM_CHECK(suite, prompt.tile_count == 0);
    CPP_LLM_CHECK(suite, prompt.visual_count == 1);
    CPP_LLM_REQUIRE(suite, prompt.chunks.size() == 3);
    CPP_LLM_CHECK(suite, prompt.chunks[0].kind == PromptChunkKind::Text);
    CPP_LLM_CHECK(suite, prompt.chunks[0].text ==
           "<|im_start|>User: \n\n<fake_token_around_image><global-img>");
    CPP_LLM_CHECK(suite, prompt.chunks[1].kind == PromptChunkKind::Visual);
    CPP_LLM_CHECK(suite, prompt.chunks[1].visual_index == 0);
    CPP_LLM_CHECK(suite, prompt.chunks[2].text ==
           "<fake_token_around_image>Describe it.<end_of_utterance>\nAssistant:");
}

void test_two_by_three_tiles(cpp_llm_engine::testing::TestSuite& suite) {
    SmolvlmPromptChunks prompt;
    std::string error;
    CPP_LLM_REQUIRE(suite, multimodal::build_smolvlm_single_turn_prompt(
        "Question?", 2, 3, &prompt, &error));
    CPP_LLM_CHECK(suite, prompt.tile_count == 6);
    CPP_LLM_CHECK(suite, prompt.visual_count == 7);
    CPP_LLM_REQUIRE(suite, prompt.chunks.size() == 15);
    for (std::size_t index = 0; index < prompt.visual_count; ++index) {
        const auto& visual = prompt.chunks[index * 2 + 1];
        CPP_LLM_CHECK(suite, visual.kind == PromptChunkKind::Visual);
        CPP_LLM_CHECK(suite, visual.visual_index == index);
    }
    CPP_LLM_CHECK(suite, prompt.chunks[0].text ==
           "<|im_start|>User: <fake_token_around_image><row_1_col_1>");
    CPP_LLM_CHECK(suite, prompt.chunks[4].text ==
           "<fake_token_around_image><row_1_col_3>");
    CPP_LLM_CHECK(suite, prompt.chunks[6].text ==
           "\n<fake_token_around_image><row_2_col_1>");
    CPP_LLM_CHECK(suite, prompt.chunks[10].text ==
           "<fake_token_around_image><row_2_col_3>");
    CPP_LLM_CHECK(suite, prompt.chunks[12].text ==
           "\n\n<fake_token_around_image><global-img>");
    CPP_LLM_CHECK(suite, prompt.chunks[14].text ==
           "<fake_token_around_image>Question?<end_of_utterance>\nAssistant:");
}

void test_invalid_inputs(cpp_llm_engine::testing::TestSuite& suite) {
    SmolvlmPromptChunks prompt;
    std::string error;
    CPP_LLM_CHECK(suite, !multimodal::build_smolvlm_single_turn_prompt(
        "", 0, 0, &prompt, &error));
    CPP_LLM_CHECK(suite, !error.empty());
    CPP_LLM_CHECK(suite, !multimodal::build_smolvlm_single_turn_prompt(
        "x", 1, 0, &prompt, &error));
    CPP_LLM_CHECK(suite, !multimodal::build_smolvlm_single_turn_prompt(
        "x", 0, 1, &prompt, &error));
    CPP_LLM_CHECK(suite, !multimodal::build_smolvlm_single_turn_prompt(
        "x", 2, std::numeric_limits<std::size_t>::max(), &prompt, &error));
    CPP_LLM_CHECK(suite, !multimodal::build_smolvlm_single_turn_prompt(
        "x", 0, 0, nullptr, &error));
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("multimodal_smolvlm_prompt");
    suite.run("overview 视觉输入展开为三个 chunk", [&] { test_overview_only(suite); });
    suite.run("2x3 tiles 按行展开且 overview 位于末尾", [&] {
        test_two_by_three_tiles(suite);
    });
    suite.run("非法问题和网格被拒绝", [&] { test_invalid_inputs(suite); });
    return suite.finish();
}
