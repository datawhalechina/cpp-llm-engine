#!/usr/bin/env bash
# Chapter 00：验证真实推理编排脚本的参数、错误分支和输出解析契约。
# 输入：临时伪模型与伪 llama-simple；输出：各分支的退出状态检查。
# 验证：不依赖网络或真实模型，适合进入默认回归与 Release 契约。
set -euo pipefail

chapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${chapter_dir}/run_real_inference.sh"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter00-contract.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

model_path="${work_dir}/model.gguf"
fake_success="${work_dir}/llama-simple-success"
fake_empty="${work_dir}/llama-simple-empty"
: >"${model_path}"

cat >"${fake_success}" <<'EOF'
#!/usr/bin/env bash
echo 'generated text'
echo 'main: decoded 2 tokens in 0.01 s, speed: 200.00 t/s' >&2
EOF
cat >"${fake_empty}" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "${fake_success}" "${fake_empty}"

run_runner() {
    CHAPTER00_LLAMA_EXECUTABLE="$1" \
    CPP_LLM_ENGINE_MODEL_PATH="$2" \
        bash "${runner}" prompt "${3:-2}"
}

expect_status() {
    local expected="$1"
    shift
    set +e
    "$@" >"${work_dir}/last.stdout" 2>"${work_dir}/last.stderr"
    local actual=$?
    set -e
    if [[ "${actual}" -ne "${expected}" ]]; then
        cat "${work_dir}/last.stdout" >&2
        cat "${work_dir}/last.stderr" >&2
        echo "expected exit ${expected}, got ${actual}" >&2
        exit 1
    fi
}

success_output="$(run_runner "${fake_success}" "${model_path}")"
grep -q 'real inference passed: decoded_tokens=2' <<<"${success_output}"

expect_status 2 run_runner "${work_dir}/missing-executable" "${model_path}"
expect_status 2 run_runner "${fake_success}" "${work_dir}/missing-model.gguf"
expect_status 2 run_runner "${fake_success}" "${model_path}" invalid
expect_status 1 run_runner "${fake_empty}" "${model_path}"

echo "chapter00 real inference script contract passed"
