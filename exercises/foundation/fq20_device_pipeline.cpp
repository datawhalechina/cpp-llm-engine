#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class DevicePipeline {
 public:
  void upload(std::vector<float> input) {
    if (stage_ != Stage::Idle && stage_ != Stage::Downloaded) {
      throw std::logic_error("previous device work is incomplete");
    }
    if (input.empty()) throw std::invalid_argument("input must not be empty");
    ensure_workspace(input.size());
    device_values_ = std::move(input);
    operations_.push_back("H2D");
    stage_ = Stage::Uploaded;
  }

  void launch_scale(float scale) {
    require(Stage::Uploaded, "kernel launch requires uploaded input");
    for (float& value : device_values_) value *= scale;
    operations_.push_back("kernel");
    stage_ = Stage::KernelQueued;
  }

  void record_event() {
    require(Stage::KernelQueued, "event requires queued kernel work");
    operations_.push_back("event");
    stage_ = Stage::EventRecorded;
  }

  void wait_event() {
    require(Stage::EventRecorded, "wait requires a recorded event");
    operations_.push_back("wait");
    stage_ = Stage::Complete;
  }

  std::vector<float> download() {
    require(Stage::Complete, "D2H requires completed kernel work");
    operations_.push_back("D2H");
    stage_ = Stage::Downloaded;
    return device_values_;
  }

  void fail_and_reset() {
    if (stage_ == Stage::Idle || stage_ == Stage::Downloaded) {
      throw std::logic_error("no in-flight work to reset");
    }
    device_values_.clear();
    operations_.push_back("reset");
    stage_ = Stage::Idle;
  }

  std::size_t workspace_capacity() const { return workspace_.size(); }
  std::size_t workspace_growths() const { return workspace_growths_; }
  const std::vector<std::string>& operations() const { return operations_; }

 private:
  enum class Stage {
    Idle,
    Uploaded,
    KernelQueued,
    EventRecorded,
    Complete,
    Downloaded
  };

  void require(Stage expected, const char* message) const {
    if (stage_ != expected) throw std::logic_error(message);
  }
  void ensure_workspace(std::size_t size) {
    if (workspace_.size() >= size) return;
    workspace_.resize(size);
    ++workspace_growths_;
  }

  Stage stage_ = Stage::Idle;
  std::vector<float> device_values_;
  std::vector<float> workspace_;
  std::vector<std::string> operations_;
  std::size_t workspace_growths_ = 0;
};

int main() {
  DevicePipeline pipeline;
  bool rejected = false;
  try {
    pipeline.launch_scale(2.0F);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  assert(rejected);

  pipeline.upload({1.0F, 2.0F, 3.0F});
  pipeline.launch_scale(2.0F);
  bool reuse_rejected = false;
  try {
    pipeline.upload({4.0F});
  } catch (const std::logic_error&) {
    reuse_rejected = true;
  }
  assert(reuse_rejected);
  pipeline.record_event();
  bool early_download_rejected = false;
  try {
    (void)pipeline.download();
  } catch (const std::logic_error&) {
    early_download_rejected = true;
  }
  assert(early_download_rejected);
  pipeline.wait_event();
  assert((pipeline.download() == std::vector<float>{2.0F, 4.0F, 6.0F}));
  assert((pipeline.operations() ==
          std::vector<std::string>{"H2D", "kernel", "event", "wait", "D2H"}));
  assert(pipeline.workspace_capacity() == 3);
  assert(pipeline.workspace_growths() == 1);

  pipeline.upload({4.0F, 5.0F});
  pipeline.launch_scale(0.5F);
  pipeline.fail_and_reset();
  assert(pipeline.workspace_capacity() == 3);
  assert(pipeline.workspace_growths() == 1);

  pipeline.upload({4.0F, 5.0F});
  pipeline.launch_scale(0.5F);
  pipeline.record_event();
  pipeline.wait_event();
  assert((pipeline.download() == std::vector<float>{2.0F, 2.5F}));
  assert(pipeline.workspace_capacity() == 3);
  assert(pipeline.workspace_growths() == 1);
  return 0;
}
