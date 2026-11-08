#include <cstddef>
#include <vector>

float read_checked(const std::vector<float>& values, std::size_t index) {
  // TODO: reject index >= values.size().
  return values[index];
}

int main() {
  // TODO: verify both valid and invalid access.
  return 1;
}
