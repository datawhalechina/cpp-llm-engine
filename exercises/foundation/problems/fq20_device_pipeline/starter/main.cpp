#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

class DevicePipeline {
 public:
  void upload(std::vector<float> input) {
    // TODO: validate the current stage, retain the input, and grow workspace.
    (void)input;
  }
  void launch_scale(float scale) {
    // TODO: require uploaded input, transform it, and advance the stage.
    (void)scale;
  }
  void record_event() {
    // TODO: require queued kernel work and record the completion boundary.
  }
  void wait_event() {
    // TODO: require a recorded event and make the output visible.
  }
  std::vector<float> download() {
    // TODO: require completed kernel work and return the transformed values.
    return {};
  }
  void fail_and_reset() {
    // TODO: clear in-flight state while retaining workspace capacity.
  }
  std::size_t workspace_capacity() const { return 0; }
  std::size_t workspace_growths() const { return 0; }
  const std::vector<std::string>& operations() const { return operations_; }

 private:
  std::vector<std::string> operations_;
};

int main() {
  DevicePipeline pipeline;
  pipeline.upload({1.0F, 2.0F, 3.0F});
  pipeline.launch_scale(2.0F);
  pipeline.record_event();
  pipeline.wait_event();
  const auto output = pipeline.download();
  pipeline.upload({4.0F, 5.0F});
  pipeline.launch_scale(0.5F);
  pipeline.fail_and_reset();
  pipeline.upload({4.0F, 5.0F});
  pipeline.launch_scale(0.5F);
  pipeline.record_event();
  pipeline.wait_event();
  const auto second_output = pipeline.download();
  const bool valid = output == std::vector<float>({2.0F, 4.0F, 6.0F}) &&
                     second_output == std::vector<float>({2.0F, 2.5F}) &&
                     pipeline.workspace_capacity() == 3 &&
                     pipeline.workspace_growths() == 1;
  if (!valid) {
    std::cerr << "complete the ordered device pipeline\n";
    return 1;
  }
  return 0;
}
