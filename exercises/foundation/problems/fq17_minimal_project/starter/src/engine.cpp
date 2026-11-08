#include "engine.h"

#include <stdexcept>

Workspace::Workspace(std::size_t size) {
  // TODO: allocate the workspace and record its size.
  (void)size;
}

Workspace::Workspace(Workspace&& other) noexcept {
  // TODO: transfer ownership and clear the source state.
  (void)other;
}

Workspace& Workspace::operator=(Workspace&& other) noexcept {
  // TODO: release the old value, transfer ownership, and clear the source.
  (void)other;
  return *this;
}

Workspace::~Workspace() = default;

int& Workspace::at(std::size_t index) {
  // TODO: reject invalid indexes and return the selected element.
  (void)index;
  throw std::logic_error("Workspace::at is not implemented");
}

std::size_t Workspace::size() const {
  // TODO: report the owned element count.
  return 0;
}

EngineStep::EngineStep(std::size_t workspace_size) : workspace_(workspace_size) {}

int EngineStep::run(int input) {
  // TODO: use the workspace and return the transformed value.
  (void)input;
  throw std::logic_error("EngineStep::run is not implemented");
}
