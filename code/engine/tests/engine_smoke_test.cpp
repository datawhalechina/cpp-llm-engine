// Engine：端到端 smoke test
// 职责：验证文本输入经过 tokenizer、Runtime 和 decode 后返回完整结果。
// 通过条件：prompt、generated、output 数量和 finished 状态符合断言。
// 验证：code/engine/scripts/run.sh。
#include "cpp_llm_engine/engine.h"

#include "cpp_llm_engine/testing/test.h"

namespace {

class FixedTokenizerBackend final : public cpp_llm_engine::TokenizerBackend {
public:
    std::size_t vocab_size() const override { return 64; }
    chapter01::TokenSequence encode(std::string_view) const override { return {7, 8}; }
    std::string decode(const chapter01::TokenSequence&) const override { return "fixed"; }
};

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("cpp_llm_engine");

    suite.run("default tokenizer completes generation", [&] {
        cpp_llm_engine::ModelConfig config;
        config.max_new_tokens = 3;
        const cpp_llm_engine::GenerationResult result =
            cpp_llm_engine::Engine(config).generate("hello");
        CPP_LLM_CHECK_EQ(suite, result.prompt_tokens.size(), 5U);
        CPP_LLM_CHECK_EQ(suite, result.generated_tokens.size(), 3U);
        CPP_LLM_CHECK_EQ(suite, result.output.size(), 3U);
        CPP_LLM_CHECK(suite, result.finished);
    });

    suite.run("token tensor preserves encoded ids", [&] {
        const chapter01::Tokenizer tokenizer(64);
        const cpp_llm_engine::TokenTensor tokens(tokenizer.encode("a!"));
        CPP_LLM_REQUIRE(suite, tokens.size() == 2);
        CPP_LLM_CHECK_EQ(suite, tokens.values()[0], 1);
        CPP_LLM_CHECK_EQ(suite, tokens.values()[1], 0);
    });

    suite.run("injected tokenizer controls prompt ids and decode", [&] {
        cpp_llm_engine::ModelConfig config;
        config.max_new_tokens = 1;
        const cpp_llm_engine::Engine engine(
            config, std::make_unique<FixedTokenizerBackend>());
        const auto result = engine.generate("real prompt");
        CPP_LLM_CHECK(suite, result.prompt_tokens == chapter01::TokenSequence({7, 8}));
        CPP_LLM_CHECK(suite, result.output == "fixed");
    });
    return suite.finish();
}
