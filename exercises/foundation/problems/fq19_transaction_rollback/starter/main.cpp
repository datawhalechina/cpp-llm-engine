#include <cstddef>
#include <iostream>
#include <vector>

class TransactionalSlots {
 public:
  explicit TransactionalSlots(std::size_t count) : values_(count, 0) {}

  bool write_pair(std::size_t first, std::size_t second, int first_value,
                  int second_value, bool fail_after_first_write) {
    (void)first;
    (void)second;
    (void)first_value;
    (void)second_value;
    (void)fail_after_first_write;
    // TODO: reserve, write, commit, and roll back both slots on failure.
    return false;
  }

  std::size_t free_count() const { return values_.size(); }
  std::size_t committed_count() const { return 0; }
  int value(std::size_t index) const { return values_.at(index); }

 private:
  std::vector<int> values_;
};

int main() {
  TransactionalSlots slots(4);
  const bool rolled_back = !slots.write_pair(0, 1, 10, 20, true) &&
                           slots.free_count() == 4 &&
                           slots.committed_count() == 0 && slots.value(0) == 0;
  const bool committed = slots.write_pair(0, 1, 10, 20, false) &&
                         slots.free_count() == 2 &&
                         slots.committed_count() == 2 && slots.value(1) == 20;
  if (!rolled_back || !committed) {
    std::cerr << "complete the transactional slot state machine\n";
    return 1;
  }
  return 0;
}
