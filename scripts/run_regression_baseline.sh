#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

run_and_check() {
    local name="$1"
    local command="$2"
    shift 2

    echo "[regression] ${name}"
    bash -c "${command}" >"${tmp_dir}/${name}.log" 2>&1
    for expected in "$@"; do
        if ! grep -Fq "${expected}" "${tmp_dir}/${name}.log"; then
            echo "[regression] missing '${expected}' in ${name}" >&2
            cat "${tmp_dir}/${name}.log" >&2
            exit 1
        fi
    done
}

run_and_check \
    code_headers \
    "cd '${repo_root}' && npm run code:check-headers" \
    "Code header check passed: code"

run_and_check \
    docs_structure \
    "cd '${repo_root}' && npm run docs:check-structure" \
    "Documentation structure check passed"

run_and_check \
    reference_assets \
    "cd '${repo_root}' && npm run docs:check-reference" \
    "Reference asset check passed"

run_and_check \
    test_assets \
    "cd '${repo_root}' && npm run test:check-assets" \
    "Test asset check passed"

run_and_check \
    foundation \
    "cd '${repo_root}' && bash exercises/foundation/run_all.sh"

run_and_check \
    foundation_starters \
    "cd '${repo_root}' && bash scripts/check_foundation_starters.sh" \
    "Foundation starter compile check passed"

run_and_check \
    foundation_metadata \
    "cd '${repo_root}' && bash scripts/check_foundation_metadata.sh" \
    "Foundation metadata check passed"

run_and_check \
    core_contracts \
    "cd '${repo_root}' && cmake -S code/core -B '${tmp_dir}/core-build' && cmake --build '${tmp_dir}/core-build' && ctest --test-dir '${tmp_dir}/core-build' --output-on-failure"

run_and_check \
    multimodal_preprocessing \
    "cd '${repo_root}' && MULTIMODAL_BUILD_DIR='${tmp_dir}/multimodal-build' bash code/multimodal/scripts/run.sh" \
    "patches=1024x768" \
    "grid=2x2" \
    "visual_images=5"

run_and_check \
    engine \
    "cd '${repo_root}' && cmake -S code/engine -B '${tmp_dir}/engine-build' && cmake --build '${tmp_dir}/engine-build' && ctest --test-dir '${tmp_dir}/engine-build' --output-on-failure && '${tmp_dir}/engine-build/cpp_llm_engine_demo' --prompt hello --max-new-tokens 2" \
    "[engine] request finished: true"

run_and_check \
    chapter01 \
    "cd '${repo_root}' && bash code/chapter01_tensor_loader/run.sh"

run_and_check \
    chapter02 \
    "cd '${repo_root}' && bash code/chapter02_tensor_ops/scripts/run.sh" \
    "shares storage: true"

run_and_check \
    chapter03 \
    "cd '${repo_root}' && bash code/chapter03_matmul/scripts/run.sh" \
    "58" \
    "154"

run_and_check \
    chapter04 \
    "cd '${repo_root}' && bash code/chapter04_attention/scripts/run.sh" \
    "ready: true"

run_and_check \
    chapter05 \
    "cd '${repo_root}' && bash code/chapter05_quantization/scripts/run.sh" \
    "quantized ready: true"

run_and_check \
    chapter06 \
    "cd '${repo_root}' && bash code/chapter06_generation/scripts/run.sh" \
    "request_a finished: true" \
    "request_b finished: true"

run_and_check \
    chapter07 \
    "cd '${repo_root}' && CHAPTER07_RESULT_DIR='${tmp_dir}/chapter07-results' bash code/chapter07_benchmark/scripts/run.sh" \
    "unit checks: true" \
    "e2e checks: true"

run_and_check \
    chapter08 \
    "cd '${repo_root}' && bash code/chapter08_profiling/scripts/run.sh" \
    "dominant_phase=" \
    "decision=KEEP"

run_and_check \
    chapter09 \
    "cd '${repo_root}' && bash code/chapter09_serving/scripts/run.sh" \
    "first_completed=2" \
    "second_completed=1" \
    "inflight=0"

echo "[regression] baseline passed"
