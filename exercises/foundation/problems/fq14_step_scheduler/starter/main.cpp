#include <cstddef>
#include <deque>
#include <vector>

struct StepResult {
  std::vector<int> request_ids;
  std::size_t tokens = 0;
};

class StepScheduler {
 public:
  void admit(int request_id, std::size_t remaining_tokens) {
    // TODO: ignore requests that have no work left.
    (void)request_id;
    (void)remaining_tokens;
  }

  StepResult run_step(std::size_t max_batch, std::size_t token_budget) {
    // TODO: choose requests in FIFO order and requeue unfinished requests.
    (void)max_batch;
    (void)token_budget;
    return {};
  }

  bool empty() const { return true; }

 private:
  struct Entry {
    int id;
    std::size_t remaining_tokens;
  };
  std::deque<Entry> queue_;
};

int main() {
  // TODO: add multi-step ordering, budget, and completion checks.
  return 1;
}
