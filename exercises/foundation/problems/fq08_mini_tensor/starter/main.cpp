#include <cstddef>
#include <memory>

class MiniTensor {
 public:
  explicit MiniTensor(std::size_t size);
  float& at(std::size_t index);
  MiniTensor slice(std::size_t offset, std::size_t size) const;

 private:
  std::shared_ptr<float[]> storage_;
  std::size_t size_ = 0;
  std::size_t offset_ = 0;
};

int main() {
  // TODO: implement shared storage, slicing, and bounds checks.
  return 1;
}
