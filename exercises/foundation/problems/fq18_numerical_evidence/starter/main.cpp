#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

double max_abs_error(const std::vector<double>& actual,
                     const std::vector<double>& expected) {
  (void)actual;
  (void)expected;
  // TODO: validate the inputs and return the largest element-wise error.
  return 0.0;
}

double root_mean_square_error(const std::vector<double>& actual,
                              const std::vector<double>& expected) {
  (void)actual;
  (void)expected;
  // TODO: return the square root of the mean squared error.
  return 0.0;
}

double cosine_similarity(const std::vector<double>& lhs,
                         const std::vector<double>& rhs) {
  (void)lhs;
  (void)rhs;
  // TODO: reject zero vectors and compare vector directions.
  return 0.0;
}

std::vector<double> stable_softmax(const std::vector<double>& logits) {
  // TODO: subtract the maximum before exponentiation and normalize the result.
  return logits;
}

int main() {
  const std::vector<double> expected{1.0, 2.0, 3.0};
  const std::vector<double> actual{1.001, 1.998, 3.001};
  const auto probabilities = stable_softmax({1000.0, 1001.0, 1002.0});
  const bool valid = std::abs(max_abs_error(actual, expected) - 0.002) < 1e-12 &&
                     root_mean_square_error(actual, expected) < 0.0015 &&
                     cosine_similarity(actual, expected) > 0.999999 &&
                     probabilities.size() == 3 && probabilities[2] > probabilities[1];
  if (!valid) {
    std::cerr << "complete the numerical evidence functions\n";
    return 1;
  }
  return 0;
}
