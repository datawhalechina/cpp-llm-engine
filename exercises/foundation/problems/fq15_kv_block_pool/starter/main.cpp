#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

class KvBlockPool {
 public:
  explicit KvBlockPool(std::size_t block_count) : free_(block_count, true) {}

  bool allocate(int request_id, std::size_t block_count) {
    // TODO: check capacity and duplicate requests before changing state.
    (void)request_id;
    (void)block_count;
    return false;
  }

  bool release(int request_id) {
    // TODO: mark the request's blocks free and erase its mapping.
    (void)request_id;
    return false;
  }

  std::size_t free_count() const { return 0; }
  std::size_t blocks_for(int request_id) const {
    (void)request_id;
    return 0;
  }

 private:
  std::vector<bool> free_;
  std::unordered_map<int, std::vector<std::size_t>> owners_;
};

int main() {
  // TODO: add allocation, rollback, release, and reuse checks.
  return 1;
}
