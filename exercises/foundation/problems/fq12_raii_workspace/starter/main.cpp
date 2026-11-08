#include <cstddef>
#include <memory>

class Workspace {
 public:
  explicit Workspace(std::size_t size);
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace(Workspace&&) noexcept;
  Workspace& operator=(Workspace&&) noexcept;
  std::size_t size() const;
  float& at(std::size_t index);

 private:
  std::unique_ptr<float[]> storage_;
  std::size_t size_ = 0;
};

int main() {
  // TODO: make workspace ownership move-only and exception-safe.
  return 1;
}
