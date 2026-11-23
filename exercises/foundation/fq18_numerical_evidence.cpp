#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

double max_abs_error(const std::vector<double>& actual,
                     const std::vector<double>& expected) {
  if (actual.size() != expected.size() || actual.empty()) {
    throw std::invalid_argument("vectors must have the same non-zero size");
  }
  double result = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    result = std::max(result, std::abs(actual[index] - expected[index]));
  }
  return result;
}

double root_mean_square_error(const std::vector<double>& actual,
                              const std::vector<double>& expected) {
  if (actual.size() != expected.size() || actual.empty()) {
    throw std::invalid_argument("vectors must have the same non-zero size");
  }
  double squared_error = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference = actual[index] - expected[index];
    squared_error += difference * difference;
  }
  return std::sqrt(squared_error / static_cast<double>(actual.size()));
}

double cosine_similarity(const std::vector<double>& lhs,
                         const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size() || lhs.empty()) {
    throw std::invalid_argument("vectors must have the same non-zero size");
  }
  double dot = 0.0;
  double lhs_norm = 0.0;
  double rhs_norm = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    dot += lhs[index] * rhs[index];
    lhs_norm += lhs[index] * lhs[index];
    rhs_norm += rhs[index] * rhs[index];
  }
  if (lhs_norm == 0.0 || rhs_norm == 0.0) {
    throw std::invalid_argument("cosine similarity needs non-zero vectors");
  }
  return dot / std::sqrt(lhs_norm * rhs_norm);
}

std::vector<double> stable_softmax(const std::vector<double>& logits) {
  if (logits.empty()) throw std::invalid_argument("logits must not be empty");
  const double maximum = *std::max_element(logits.begin(), logits.end());
  std::vector<double> probabilities(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    probabilities[index] = std::exp(logits[index] - maximum);
  }
  const double sum =
      std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
  for (double& probability : probabilities) probability /= sum;
  return probabilities;
}

int main() {
  const std::vector<double> expected{1.0, 2.0, 3.0};
  const std::vector<double> actual{1.001, 1.998, 3.001};
  assert(std::abs(max_abs_error(actual, expected) - 0.002) < 1e-12);
  assert(root_mean_square_error(actual, expected) < 0.0015);
  assert(cosine_similarity(actual, expected) > 0.999999);

  const auto probabilities = stable_softmax({1000.0, 1001.0, 1002.0});
  assert(probabilities[2] > probabilities[1]);
  assert(probabilities[1] > probabilities[0]);
  assert(std::abs(std::accumulate(probabilities.begin(), probabilities.end(), 0.0) -
                  1.0) < 1e-12);

  bool rejected = false;
  try {
    (void)max_abs_error({1.0}, {1.0, 2.0});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  bool rmse_rejected = false;
  try {
    (void)root_mean_square_error({}, {});
  } catch (const std::invalid_argument&) {
    rmse_rejected = true;
  }
  assert(rmse_rejected);
  bool cosine_rejected = false;
  try {
    (void)cosine_similarity({0.0, 0.0}, {1.0, 2.0});
  } catch (const std::invalid_argument&) {
    cosine_rejected = true;
  }
  assert(cosine_rejected);
  bool softmax_rejected = false;
  try {
    (void)stable_softmax({});
  } catch (const std::invalid_argument&) {
    softmax_rejected = true;
  }
  assert(softmax_rejected);
  return 0;
}
