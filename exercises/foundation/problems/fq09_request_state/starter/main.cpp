#include <cstddef>
#include <vector>

enum class RequestPhase { Waiting, Prefill, Running, Finished, Aborted };

class RequestState {
 public:
  RequestState(std::size_t prompt_size, std::size_t max_new_tokens);
  RequestPhase phase() const;
  void admit();
  void finish_prefill();
  bool append_generated(int token);
  void abort();

 private:
  std::size_t prompt_size_ = 0;
  std::size_t max_new_tokens_ = 0;
  std::vector<int> generated_;
  RequestPhase phase_ = RequestPhase::Waiting;
};

int main() {
  // TODO: enforce legal runtime transitions and terminal cleanup rules.
  return 1;
}
