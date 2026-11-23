#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

std::vector<float> matmul(const std::vector<float>& a, std::size_t m, std::size_t k,
                          const std::vector<float>& b, std::size_t n) {
  if (a.size() != m * k || b.size() != k * n) throw std::invalid_argument("shape mismatch");
  std::vector<float> c(m * n, 0.0F);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      for (std::size_t inner = 0; inner < k; ++inner) {
        c[row * n + col] += a[row * k + inner] * b[inner * n + col];
      }
    }
  }
  return c;
}

int main() {
  assert(matmul({2}, 1, 1, {3}, 1) == std::vector<float>({6}));
  assert(matmul({1, 2, 3}, 1, 3, {1, 0, 0, 1, 1, 1}, 2) ==
         std::vector<float>({4, 5}));
  assert(matmul({1, 2, 3}, 3, 1, {4}, 1) ==
         std::vector<float>({4, 8, 12}));
  const auto result = matmul({1, 2, 3, 4}, 2, 2, {5, 6, 7, 8}, 2);
  assert(result == std::vector<float>({19, 22, 43, 50}));
  bool rejected = false;
  try { (void)matmul({1, 2}, 2, 2, {1}, 1); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
