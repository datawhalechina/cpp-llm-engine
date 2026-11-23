#include <cassert>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

std::size_t numel(const std::vector<std::size_t>& shape) {
  if (shape.empty()) throw std::invalid_argument("shape must not be empty");
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, [](std::size_t total, std::size_t dim) {
    if (dim == 0) throw std::invalid_argument("dimension must be positive");
    if (total > std::numeric_limits<std::size_t>::max() / dim) {
      throw std::overflow_error("shape product overflow");
    }
    return total * dim;
  });
}

int main() {
  assert(numel({2, 3, 4}) == 24);
  assert(numel({1}) == 1);
  bool zero_rejected = false;
  try { (void)numel({2, 0}); } catch (const std::invalid_argument&) { zero_rejected = true; }
  assert(zero_rejected);
  bool empty_rejected = false;
  try { (void)numel({}); } catch (const std::invalid_argument&) { empty_rejected = true; }
  assert(empty_rejected);
  bool overflow_rejected = false;
  try {
    (void)numel({std::numeric_limits<std::size_t>::max(), 2});
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  assert(overflow_rejected);
}
