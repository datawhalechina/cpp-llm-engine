#include <cstddef>
#include <deque>
#include <optional>

class RequestQueue {
 public:
  explicit RequestQueue(std::size_t capacity);
  bool admit(int request_id);
  bool cancel(int request_id);
  std::optional<int> pop();
  std::size_t size() const;

 private:
  std::size_t capacity_ = 0;
  std::deque<int> queue_;
};

int main() {
  // TODO: enforce capacity, FIFO order, cancellation, and empty-queue rules.
  return 1;
}
