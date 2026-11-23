#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

enum class TerminalStatus { Completed, Failed, Cancelled };

struct Completion {
  int request_id = 0;
  TerminalStatus status = TerminalStatus::Failed;
  int value = 0;
};

class BoundedWorker {
 public:
  explicit BoundedWorker(std::size_t capacity) { (void)capacity; }
  bool submit(int request_id, std::function<int()> work) {
    (void)request_id;
    (void)work;
    // TODO: enforce capacity and queue the request under a mutex.
    return false;
  }
  bool cancel(int request_id) {
    (void)request_id;
    // TODO: remove a pending task and record one cancelled terminal.
    return false;
  }
  void start() {
    // TODO: start one worker thread that waits on a condition variable.
  }
  void stop() {
    // TODO: stop admission, drain accepted work, notify, and join.
  }
  void wait_for_terminals(std::size_t count) { (void)count; }
  std::vector<Completion> completions() { return {}; }
};

int main() {
  BoundedWorker worker(2);
  const bool accepted = worker.submit(10, [] { return 11; });
  const bool cancelled = worker.submit(11, [] { return 12; }) && worker.cancel(11);
  worker.start();
  worker.wait_for_terminals(2);
  worker.stop();
  const auto completions = worker.completions();
  if (!accepted || !cancelled || completions.size() != 2) {
    std::cerr << "complete the bounded worker lifecycle\n";
    return 1;
  }
  return 0;
}
