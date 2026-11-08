#include <cstddef>

struct TensorView {
  float* data = nullptr;
  std::size_t length = 0;
  std::size_t offset = 0;

  float& at(std::size_t index) {
    // TODO: validate index and return the offset element.
    return data[index];
  }
};

int main() {
  // TODO: create storage, create a view, and verify shared updates.
  return 1;
}
