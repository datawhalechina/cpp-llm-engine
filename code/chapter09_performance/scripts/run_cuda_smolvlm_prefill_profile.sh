#!/usr/bin/env bash
# Chapter09：采集 SmolVLM 双请求 Packed Prefill 的 CUDA 计算核分布。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_SMOLVLM_PREFILL_PROFILE_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-smolvlm-prefill-profile-build}"
result_dir="${CHAPTER09_SMOLVLM_PREFILL_PROFILE_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-smolvlm-prefill-profile}"
mmproj="${SMOLVLM_MMPROJ_PATH:-${repo_root}/llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf}"
model="${SMOLVLM_MODEL_PATH:-${repo_root}/llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf}"
image="${SMOLVLM_IMAGE_PATH:-${repo_root}/llama.cpp/tools/mtmd/test-1.jpeg}"
question="${SMOLVLM_QUESTION:-What historic event is shown? Answer in one short sentence.}"
profile_base="${result_dir}/smolvlm-packed-prefill"

if [[ ! -x "${cuda_home}/bin/nsys" ]]; then
    echo "nsys not found under ${cuda_home}" >&2
    exit 2
fi
mkdir -p "${result_dir}"

CUDA_HOME="${cuda_home}" \
MULTIMODAL_CUDA_BUILD_DIR="${build_dir}" \
bash "${repo_root}/code/multimodal/scripts/run_cuda_smolvlm_context.sh" \
    "${mmproj}" "${model}" "${image}" mixed-service "${question}" \
    >"${result_dir}/build-and-smoke.txt"
grep -q '^service_packed_prefill_calls: 1$' \
    "${result_dir}/build-and-smoke.txt"

"${cuda_home}/bin/nsys" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="${profile_base}" \
    "${build_dir}/multimodal_cuda_smolvlm_context_inspect" \
    "${mmproj}" "${model}" "${image}" --mixed-service "${question}" \
    >"${result_dir}/program-output.txt"
grep -q '^service_packed_prefill_calls: 1$' \
    "${result_dir}/program-output.txt"

"${cuda_home}/bin/nsys" stats \
    --force-export=true \
    --report cuda_gpu_kern_sum \
    --format csv \
    "${profile_base}.nsys-rep" \
    >"${result_dir}/kernel-summary.csv"

cat >"${result_dir}/metadata.txt" <<EOF
experiment=smolvlm-cuda-packed-prefill-profile
workload=two identical multimodal requests in mixed-service mode
model_asset=${model##*/}
mmproj_asset=${mmproj##*/}
image_asset=${image##*/}
profiler=Nsight Systems
correctness=PASS
EOF
"${cuda_home}/bin/nsys" --version >>"${result_dir}/metadata.txt"

echo "smolvlm_cuda_prefill_profile_result_dir=${result_dir}"
echo "smolvlm_cuda_prefill_profile=PASS"
