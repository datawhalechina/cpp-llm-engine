// Chapter 07：运行固定 trace、hotspot 和 benchmark summary 演示。
// 职责：展示结构化性能证据的输出格式，不代表真实模型测量结果。
// 验证：依赖 Chapter 07 test helpers 的 unit/e2e 检查并打印可读结果。
#include "chapter07_benchmark/benchmark.h"
#include "chapter07_benchmark/test_helpers.h"

#include <iostream>

int main() {
    const auto result = chapter07_benchmark::make_result("demo", 100, 25.0);
    const auto samples = chapter07_benchmark::sample_generation_path();

    std::cout << "[chapter07] benchmark demo\n";
    std::cout << "unit checks: " << std::boolalpha << chapter07_benchmark::run_unit_checks() << '\n';
    std::cout << "e2e checks: " << std::boolalpha << chapter07_benchmark::run_e2e_checks() << '\n';
    std::cout << "result name: " << result.name << '\n';
    std::cout << "result items/s: " << result.items_per_second << '\n';
    std::cout << "samples: " << samples.size() << '\n';
    std::cout << chapter07_benchmark::render_summary();
    std::cout << chapter07_benchmark::render_trace();
    std::cout << chapter07_benchmark::render_hotspots();
    std::cout << chapter07_benchmark::render_trace_json();
    std::cout << chapter07_benchmark::render_chrome_trace_json();
    std::cout << chapter07_benchmark::render_multi_request_trace_json();
    return 0;
}
