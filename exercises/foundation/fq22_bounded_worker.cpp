#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

enum class TerminalStatus { Completed, Failed, Cancelled };

struct Completion {
  int request_id = 0;
  TerminalStatus status = TerminalStatus::Failed;
  int value = 0;
};

class BoundedWorker {
 public:
  explicit BoundedWorker(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) throw std::invalid_argument("capacity must not be zero");
  }

  ~BoundedWorker() { stop(); }
  BoundedWorker(const BoundedWorker&) = delete;
  BoundedWorker& operator=(const BoundedWorker&) = delete;

  bool submit(int request_id, std::function<int()> work) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || queue_.size() >= capacity_ || active_ids_.count(request_id) != 0) {
      return false;
    }
    queue_.push_back({request_id, std::move(work)});
    active_ids_.insert(request_id);
    work_ready_.notify_one();
    return true;
  }

  bool cancel(int request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto task = std::find_if(queue_.begin(), queue_.end(),
                                   [request_id](const Task& value) {
                                     return value.request_id == request_id;
                                   });
    if (task == queue_.end()) return false;
    queue_.erase(task);
    active_ids_.erase(request_id);
    completions_.push_back({request_id, TerminalStatus::Cancelled, 0});
    terminal_ready_.notify_all();
    return true;
  }

  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_ || stopping_) throw std::logic_error("worker cannot start");
    started_ = true;
    worker_ = std::thread([this] { run(); });
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      work_ready_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
  }

  void wait_for_terminals(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    terminal_ready_.wait(lock, [this, count] { return completions_.size() >= count; });
  }

  template <class Rep, class Period>
  bool wait_for_terminals_for(
      std::size_t count,
      const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return terminal_ready_.wait_for(
        lock, timeout, [this, count] { return completions_.size() >= count; });
  }

  std::vector<Completion> completions() {
    std::lock_guard<std::mutex> lock(mutex_);
    return completions_;
  }

 private:
  struct Task {
    int request_id;
    std::function<int()> work;
  };

  void run() {
    while (true) {
      Task task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (stopping_ && queue_.empty()) return;
        task = std::move(queue_.front());
        queue_.pop_front();
      }

      Completion completion{task.request_id, TerminalStatus::Completed, 0};
      try {
        completion.value = task.work();
      } catch (...) {
        completion.status = TerminalStatus::Failed;
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ids_.erase(task.request_id);
        completions_.push_back(completion);
        terminal_ready_.notify_all();
      }
    }
  }

  const std::size_t capacity_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable terminal_ready_;
  std::deque<Task> queue_;
  std::unordered_set<int> active_ids_;
  std::vector<Completion> completions_;
  std::thread worker_;
  bool started_ = false;
  bool stopping_ = false;
};

int main() {
  BoundedWorker worker(3);
  assert(worker.submit(10, [] { return 11; }));
  assert(worker.submit(11, [] { return 12; }));
  assert(worker.submit(12, []() -> int { throw std::runtime_error("failed"); }));
  assert(!worker.submit(10, [] { return 0; }));
  assert(!worker.submit(13, [] { return 14; }));
  assert(worker.cancel(11));
  assert(worker.submit(13, [] { return 14; }));

  worker.start();
  worker.wait_for_terminals(4);
  assert(worker.wait_for_terminals_for(4, std::chrono::milliseconds(1)));
  assert(!worker.wait_for_terminals_for(5, std::chrono::milliseconds(5)));
  worker.stop();
  assert(!worker.submit(14, [] { return 15; }));
  assert(!worker.cancel(99));

  const auto completions = worker.completions();
  assert(completions.size() == 4);
  for (int request_id : {10, 11, 12, 13}) {
    assert(std::count_if(completions.begin(), completions.end(),
                         [request_id](const Completion& completion) {
                           return completion.request_id == request_id;
                         }) == 1);
  }
  const auto status_for = [&completions](int request_id) {
    return std::find_if(completions.begin(), completions.end(),
                        [request_id](const Completion& completion) {
                          return completion.request_id == request_id;
                        })->status;
  };
  assert(status_for(10) == TerminalStatus::Completed);
  assert(status_for(11) == TerminalStatus::Cancelled);
  assert(status_for(12) == TerminalStatus::Failed);
  assert(status_for(13) == TerminalStatus::Completed);
  return 0;
}
