#include <cstddef>
#include <memory>

class Buffer {
 public:
  explicit Buffer(std::size_t size);
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  std::size_t size() const;
  float* data();

 private:
  std::size_t size_ = 0;
  std::unique_ptr<float[]> data_;
};

int main() {
  // TODO: implement Buffer and add move-semantics assertions.
  return 1;
}
