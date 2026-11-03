// Chapter 06：以表驱动方式验证 Request 状态机和终态吸收契约。
// 职责：覆盖公共状态转换组合，而不是只复现 Runtime 的正常调用路径。
// 验证入口：chapter06_request_contract_test 或带 contract 标签的 CTest。
#include "chapter06_generation/request.h"

#include "cpp_llm_engine/testing/test.h"

#include <array>
#include <functional>
#include <stdexcept>
#include <string>

namespace {

using chapter06_generation::Request;
using chapter06_generation::RequestPhase;

struct TerminalTransition {
    const char* name;
    RequestPhase expected;
    std::function<void(Request&)> enter;
};

const std::array<TerminalTransition, 5> terminal_transitions = {{
    {"finished", RequestPhase::Finished, [](Request& request) { request.mark_finished(); }},
    {"cancelled", RequestPhase::Cancelled, [](Request& request) { request.cancel(); }},
    {"timed_out", RequestPhase::TimedOut, [](Request& request) { request.mark_timed_out(); }},
    {"rejected", RequestPhase::Rejected, [](Request& request) { request.reject(); }},
    {"failed", RequestPhase::Failed, [](Request& request) { request.mark_failed("failure"); }},
}};

void try_all_transitions(Request& request) {
    request.advance_prefill(1);
    request.append_generated_token(9);
    request.mark_prefilled();
    request.mark_decoding();
    request.mark_finished();
    request.cancel();
    request.mark_timed_out();
    request.reject();
    request.mark_failed("second failure");
}

}  // namespace

int main() {
    cpp_llm_engine::testing::TestSuite suite("chapter06_request_contract");

    suite.run("request exposes prompt and owns cache independently", [&] {
        Request request({1, 2, 3});
        CPP_LLM_CHECK(suite, request.prompt_size() == 3);
        CPP_LLM_CHECK(suite, request.prompt() == chapter06_generation::TokenSequence({1, 2, 3}));
        CPP_LLM_CHECK(suite, request.phase() == RequestPhase::Initial);
        request.cancel();
        request.cache().append(9);
        CPP_LLM_CHECK(suite, request.cancelled());
        CPP_LLM_CHECK(suite, request.cache().size() == 1);
    });

    for (const auto& transition : terminal_transitions) {
        suite.run(std::string("terminal state is absorbing: ") + transition.name, [&] {
            Request request({1, 2, 3});
            transition.enter(request);
            const auto failure_reason = request.failure_reason();

            try_all_transitions(request);

            CPP_LLM_CHECK(suite, request.phase() == transition.expected);
            CPP_LLM_CHECK(suite, request.terminal());
            CPP_LLM_CHECK(suite, request.prefill_offset() == 0);
            CPP_LLM_CHECK(suite, request.generated().empty());
            CPP_LLM_CHECK(suite, request.failure_reason() == failure_reason);
        });
    }

    suite.run("prefill progress is monotonic and bounded", [&] {
        Request request({1, 2, 3});
        request.advance_prefill(1);
        CPP_LLM_CHECK(suite, request.prefill_offset() == 1);
        CPP_LLM_CHECK(suite, request.remaining_prompt_tokens() == 2);
        request.advance_prefill(2);
        CPP_LLM_CHECK(suite, request.prefill_offset() == 3);
        CPP_LLM_CHECK(suite, request.remaining_prompt_tokens() == 0);

        bool rejected = false;
        try {
            request.advance_prefill(1);
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        CPP_LLM_CHECK(suite, rejected);
        CPP_LLM_CHECK(suite, request.prefill_offset() == 3);
    });

    return suite.finish();
}
