#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep the smoke check independent from build caches copied from older paths.
build_dir="${CHAPTER00_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-chapter00.XXXXXX)}"
cleanup_build=false
if [[ -z "${CHAPTER00_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${script_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
"${build_dir}/chapter00_quickstart" --check
