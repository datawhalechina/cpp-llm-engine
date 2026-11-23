#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

class TransactionalSlots {
 public:
  explicit TransactionalSlots(std::size_t count)
      : states_(count, State::Free), values_(count, 0) {
    if (count == 0) throw std::invalid_argument("slot count must not be zero");
  }

  bool write_pair(std::size_t first, std::size_t second, int first_value,
                  int second_value, bool fail_after_first_write) {
    if (first == second || first >= states_.size() || second >= states_.size() ||
        states_[first] != State::Free || states_[second] != State::Free) {
      return false;
    }

    reserve(first);
    reserve(second);
    values_[first] = first_value;
    if (fail_after_first_write) {
      rollback(first);
      rollback(second);
      return false;
    }
    values_[second] = second_value;
    commit(first);
    commit(second);
    return true;
  }

  bool release(std::size_t index) {
    if (index >= states_.size() || states_[index] != State::Committed) return false;
    states_[index] = State::Free;
    values_[index] = 0;
    return true;
  }

  std::size_t free_count() const { return count(State::Free); }
  std::size_t reserved_count() const { return count(State::Reserved); }
  std::size_t committed_count() const { return count(State::Committed); }
  int value(std::size_t index) const { return values_.at(index); }

 private:
  enum class State { Free, Reserved, Committed };

  void reserve(std::size_t index) { states_[index] = State::Reserved; }
  void commit(std::size_t index) { states_[index] = State::Committed; }
  void rollback(std::size_t index) {
    states_[index] = State::Free;
    values_[index] = 0;
  }
  std::size_t count(State state) const {
    std::size_t result = 0;
    for (State current : states_) {
      if (current == state) ++result;
    }
    return result;
  }

  std::vector<State> states_;
  std::vector<int> values_;
};

int main() {
  TransactionalSlots slots(4);
  assert(!slots.write_pair(0, 1, 10, 20, true));
  assert(slots.free_count() == 4);
  assert(slots.reserved_count() == 0);
  assert(slots.committed_count() == 0);
  assert(slots.value(0) == 0);

  assert(slots.write_pair(0, 1, 10, 20, false));
  assert(slots.free_count() == 2);
  assert(slots.committed_count() == 2);
  assert(slots.value(0) == 10);
  assert(slots.value(1) == 20);
  assert(!slots.write_pair(0, 2, 30, 40, false));
  assert(slots.free_count() == 2);

  assert(slots.release(0));
  assert(slots.release(1));
  assert(slots.free_count() == 4);
  assert(!slots.release(1));
  return 0;
}
