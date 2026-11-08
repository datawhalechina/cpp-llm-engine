#include <cstddef>
#include <stdexcept>
#include <vector>

class BatchLayout {
 public:
  explicit BatchLayout(std::size_t capacity) : capacity_(capacity) {}

  bool add(std::size_t sequence_length) {
    // TODO: reject invalid lengths and requests beyond capacity.
    (void)sequence_length;
    return false;
  }

  std::size_t batch_size() const { return lengths_.size(); }
  std::size_t max_sequence_length() const { return 0; }
  std::size_t padded_tokens() const { return 0; }
  std::size_t active_tokens() const { return 0; }
  std::size_t row_offset(std::size_t row) const {
    if (row >= batch_size()) throw std::out_of_range("batch row");
    return 0;
  }

 private:
  std::size_t capacity_;
  std::vector<std::size_t> lengths_;
};

int main() {
  // TODO: add layout, capacity, padding, and invalid-input checks.
  return 1;
}
