#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

struct PairedBenchmarkResult {
  std::vector<double> baseline_ms;
  std::vector<double> candidate_ms;
  std::vector<bool> candidate_first;
  double baseline_median_ms = 0.0;
  double candidate_median_ms = 0.0;
};

PairedBenchmarkResult run_paired_benchmark(
    const std::function<double()>& baseline,
    const std::function<double()>& candidate, std::size_t warmup,
    std::size_t iterations, double tolerance) {
  (void)baseline;
  (void)candidate;
  (void)warmup;
  (void)iterations;
  (void)tolerance;
  // TODO: warm up, alternate order, check correctness, and compute medians.
  return {};
}

int main() {
  std::size_t baseline_calls = 0;
  std::size_t candidate_calls = 0;
  const auto baseline = [&baseline_calls] {
    ++baseline_calls;
    return 42.0;
  };
  const auto candidate = [&candidate_calls] {
    ++candidate_calls;
    return 42.0;
  };
  const auto result = run_paired_benchmark(baseline, candidate, 2, 5, 1e-12);
  const bool valid = result.baseline_ms.size() == 5 &&
                     result.candidate_ms.size() == 5 &&
                     baseline_calls == 7 && candidate_calls == 7 &&
                     result.candidate_first ==
                         std::vector<bool>({false, true, false, true, false});
  if (!valid) {
    std::cerr << "complete the paired benchmark harness\n";
    return 1;
  }
  return 0;
}
