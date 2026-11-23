#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

struct PairedBenchmarkResult {
  std::vector<double> baseline_ms;
  std::vector<double> candidate_ms;
  std::vector<bool> candidate_first;
  double baseline_median_ms = 0.0;
  double candidate_median_ms = 0.0;
};

double median(std::vector<double> values) {
  if (values.empty()) throw std::invalid_argument("samples must not be empty");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) return values[middle];
  return (values[middle - 1] + values[middle]) / 2.0;
}

PairedBenchmarkResult run_paired_benchmark(
    const std::function<double()>& baseline,
    const std::function<double()>& candidate, std::size_t warmup,
    std::size_t iterations, double tolerance) {
  if (iterations == 0) throw std::invalid_argument("iterations must not be zero");
  if (tolerance < 0.0) throw std::invalid_argument("tolerance must not be negative");
  for (std::size_t index = 0; index < warmup; ++index) {
    (void)baseline();
    (void)candidate();
  }

  PairedBenchmarkResult result;
  auto measure = [](const std::function<double()>& workload) {
    const auto start = std::chrono::steady_clock::now();
    const double value = workload();
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(stop - start).count();
    return std::pair<double, double>{value, elapsed};
  };

  for (std::size_t index = 0; index < iterations; ++index) {
    const bool candidate_first = index % 2 == 1;
    result.candidate_first.push_back(candidate_first);
    std::pair<double, double> baseline_sample;
    std::pair<double, double> candidate_sample;
    if (candidate_first) {
      candidate_sample = measure(candidate);
      baseline_sample = measure(baseline);
    } else {
      baseline_sample = measure(baseline);
      candidate_sample = measure(candidate);
    }
    if (std::abs(baseline_sample.first - candidate_sample.first) > tolerance) {
      throw std::runtime_error("candidate failed the correctness gate");
    }
    result.baseline_ms.push_back(baseline_sample.second);
    result.candidate_ms.push_back(candidate_sample.second);
  }
  result.baseline_median_ms = median(result.baseline_ms);
  result.candidate_median_ms = median(result.candidate_ms);
  return result;
}

int main() {
  std::size_t baseline_calls = 0;
  std::size_t candidate_calls = 0;
  const auto baseline = [&baseline_calls]() {
    ++baseline_calls;
    double sum = 0.0;
    for (int index = 0; index < 1000; ++index) sum += index * 0.5;
    return sum;
  };
  const auto candidate = [&candidate_calls]() {
    ++candidate_calls;
    double sum = 0.0;
    for (int index = 0; index < 1000; ++index) sum += index * 0.5;
    return sum;
  };
  const auto result = run_paired_benchmark(baseline, candidate, 2, 5, 1e-12);
  assert(baseline_calls == 7);
  assert(candidate_calls == 7);
  assert(result.baseline_ms.size() == 5);
  assert(result.candidate_ms.size() == 5);
  assert((result.candidate_first ==
          std::vector<bool>{false, true, false, true, false}));
  assert(result.baseline_median_ms >= 0.0);
  assert(result.candidate_median_ms >= 0.0);
  assert(median({4.0, 1.0, 3.0, 2.0}) == 2.5);

  bool rejected = false;
  try {
    (void)run_paired_benchmark([] { return 1.0; }, [] { return 2.0; }, 0, 1,
                               0.01);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  bool negative_tolerance_rejected = false;
  try {
    (void)run_paired_benchmark([] { return 1.0; }, [] { return 1.0; }, 0, 1,
                               -0.01);
  } catch (const std::invalid_argument&) {
    negative_tolerance_rejected = true;
  }
  assert(negative_tolerance_rejected);
  return 0;
}
