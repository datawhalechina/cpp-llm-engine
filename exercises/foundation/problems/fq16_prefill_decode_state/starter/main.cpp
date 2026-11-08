#include <cstddef>
#include <stdexcept>

enum class Phase { Waiting, Prefill, Decode, Finished, Aborted };

class GenerationState {
 public:
  explicit GenerationState(std::size_t max_new_tokens) : max_new_tokens_(max_new_tokens) {}

  void admit() { /* TODO */ }
  void finish_prefill() { /* TODO */ }
  bool append_decode_token() { return false; }
  void abort() { /* TODO */ }
  Phase phase() const { return phase_; }
  std::size_t generated_tokens() const { return generated_tokens_; }

 private:
  std::size_t max_new_tokens_;
  std::size_t generated_tokens_ = 0;
  Phase phase_ = Phase::Waiting;
};

int main() {
  // TODO: add legal transition, completion, cancellation, and failure checks.
  return 1;
}
