// 测试支持：按命名故障点和调用次数确定性地触发失败。
// 职责：统一组件测试中的失败时机，不进入生产对象或运行时配置。
#pragma once

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace cpp_llm_engine::testing {

enum class FaultMode {
    once,
    persistent,
};

class FaultInjector final {
public:
    void arm(
        std::string point,
        std::size_t fail_on_call = 1,
        FaultMode mode = FaultMode::once) {
        if (point.empty()) throw std::invalid_argument("fault point must not be empty");
        if (fail_on_call == 0) {
            throw std::invalid_argument("fault call index must be positive");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        rules_[std::move(point)] = Rule{fail_on_call, mode, 0, 0};
    }

    bool hit(std::string_view point) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = rules_.find(std::string(point));
        if (found == rules_.end()) return false;

        Rule& rule = found->second;
        ++rule.calls;
        const bool should_fail =
            rule.mode == FaultMode::persistent
                ? rule.calls >= rule.fail_on_call
                : rule.calls == rule.fail_on_call;
        if (should_fail) ++rule.failures;
        return should_fail;
    }

    bool armed(std::string_view point) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return rules_.find(std::string(point)) != rules_.end();
    }

    std::size_t calls(std::string_view point) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = rules_.find(std::string(point));
        return found == rules_.end() ? 0 : found->second.calls;
    }

    std::size_t failures(std::string_view point) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = rules_.find(std::string(point));
        return found == rules_.end() ? 0 : found->second.failures;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        rules_.clear();
    }

private:
    struct Rule {
        std::size_t fail_on_call;
        FaultMode mode;
        std::size_t calls;
        std::size_t failures;
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Rule> rules_;
};

}  // namespace cpp_llm_engine::testing
