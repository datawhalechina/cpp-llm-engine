// Chapter 09：验证 transport-neutral ResponseWriter 的最小生命周期。
// 职责：模拟网络写出、客户端断开和背压，不启动 socket，也不修改 ServiceCore 状态。
// 验证入口：Chapter 09 CTest target 或 scripts/run.sh。
#include "chapter09_serving/response_writer.h"

#include "cpp_llm_engine/testing/test.h"

#include <string>

namespace {

class FakeWriter final : public chapter09_serving::ResponseWriter {
public:
    bool write_headers(const chapter09_serving::ResponseHead& head) override {
        if (!connected_ || finished_) return false;
        head_ = head;
        headers_written_ = true;
        return true;
    }

    chapter09_serving::WriteResult write_event(std::string_view event) override {
        if (!connected_) return chapter09_serving::WriteResult::ClientClosed;
        if (blocked_) return chapter09_serving::WriteResult::Backpressure;
        if (finished_ || !headers_written_) return chapter09_serving::WriteResult::ClientClosed;
        body_ += event;
        return chapter09_serving::WriteResult::Ok;
    }

    void finish() override { finished_ = true; }

    bool client_connected() const override { return connected_; }

    void disconnect() { connected_ = false; }
    void set_blocked(bool blocked) { blocked_ = blocked; }

    const chapter09_serving::ResponseHead& head() const { return head_; }
    const std::string& body() const { return body_; }
    bool finished() const { return finished_; }

private:
    chapter09_serving::ResponseHead head_;
    std::string body_;
    bool connected_ = true;
    bool blocked_ = false;
    bool headers_written_ = false;
    bool finished_ = false;
};

}  // namespace

int main() {
    using namespace chapter09_serving;
    cpp_llm_engine::testing::TestSuite suite("chapter09_response_writer");

    suite.run("writes headers and event while connected", [&] {
        FakeWriter writer;
        CPP_LLM_CHECK(suite, writer.write_headers({200, "text/event-stream"}));
        CPP_LLM_CHECK_EQ(suite, writer.head().status_code, 200);
        CPP_LLM_CHECK(suite, writer.write_event("token-1\n\n") == WriteResult::Ok);
        CPP_LLM_CHECK(suite, writer.body() == "token-1\n\n");
    });

    suite.run("backpressure preserves buffered body", [&] {
        FakeWriter writer;
        CPP_LLM_REQUIRE(suite, writer.write_headers({200, "text/event-stream"}));
        CPP_LLM_REQUIRE(suite, writer.write_event("token-1\n\n") == WriteResult::Ok);
        writer.set_blocked(true);
        CPP_LLM_CHECK(
            suite, writer.write_event("token-2\n\n") == WriteResult::Backpressure);
        CPP_LLM_CHECK(suite, writer.body() == "token-1\n\n");
    });

    suite.run("disconnect and finish reject further writes", [&] {
        FakeWriter writer;
        CPP_LLM_REQUIRE(suite, writer.write_headers({200, "text/event-stream"}));
        writer.disconnect();
        CPP_LLM_CHECK(suite, !writer.client_connected());
        CPP_LLM_CHECK(
            suite, writer.write_event("token-3\n\n") == WriteResult::ClientClosed);

        writer.finish();
        CPP_LLM_CHECK(suite, writer.finished());
        CPP_LLM_CHECK(suite, !writer.write_headers({200, "text/event-stream"}));
    });
    return suite.finish();
}
