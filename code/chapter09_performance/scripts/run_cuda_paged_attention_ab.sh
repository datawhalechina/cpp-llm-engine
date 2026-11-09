#!/usr/bin/env bash
# Chapter 09：采集分页 K/V 下三阶段与融合在线 Attention 的真实 A/B 样本。
# 产物：环境元数据和两轮独立程序输出；默认写入临时目录。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
model_path="${1:-${CPP_LLM_ENGINE_MODEL_PATH:-${repo_root}/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CHAPTER09_CUDA_PAGED_ATTENTION_BUILD_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-paged-attention-build}"
result_dir="${CHAPTER09_CUDA_PAGED_ATTENTION_RESULT_DIR:-${TMPDIR:-/tmp}/cpp-llm-chapter09-cuda-paged-attention-result}"

for command in cmake nvidia-smi; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "required command not found: ${command}" >&2
        exit 2
    }
done
if [[ ! -x "${cuda_home}/bin/nvcc" || ! -f "${model_path}" ]]; then
    echo "CUDA toolkit or model is missing" >&2
    exit 2
fi

mkdir -p "${build_dir}" "${result_dir}"
cmake -S "${repo_root}/code/engine" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCPP_LLM_ENGINE_ENABLE_CUDA=ON \
    -DCHAPTER05_SOURCE_DIR="${repo_root}/code/chapter05_quantization" \
    -DCPP_LLM_ENGINE_CUDA_REAL_MODEL="${model_path}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
    --target real_tinyllama_cuda_paged_attention_benchmark --parallel 2

cat >"${result_dir}/metadata.txt" <<EOF
experiment_id=EXP-GPU-07
workload_id=WL-GPU-TINYLLAMA-PAGED-ATTENTION-AB-V1
model=TinyLlama-1.1B-Chat-v1.0.Q4_K_M
backend=CUDA
build_type=Release
prompt_tokens=32
decode_steps=1
block_size=8
warmup=2
iterations=10
independent_runs=2
statistic=median
baseline=paged-three-stage-attention
experiment=paged-fused-online-attention
timing_scope=excludes-model-load-context-create-and-request-bind
EOF
"${cuda_home}/bin/nvcc" --version >>"${result_dir}/metadata.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader >>"${result_dir}/metadata.txt"
for run in 1 2; do
    "${build_dir}/real_tinyllama_cuda_paged_attention_benchmark" "${model_path}" \
        >"${result_dir}/run-${run}.txt"
done
if [[ "${CHAPTER09_CUDA_PAGED_ATTENTION_PROFILE:-0}" == "1" ]]; then
    nsys_bin="$(command -v nsys || true)"
    if [[ -z "${nsys_bin}" ]]; then
        echo "nsys is required when profiling is enabled" >&2
        exit 2
    fi
    profile_base="${result_dir}/paged-attention-profile"
    nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu \
        --format=csv,noheader >"${result_dir}/gpu-state-before.txt"
    "${nsys_bin}" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --force-overwrite=true --output="${profile_base}" \
        "${build_dir}/real_tinyllama_cuda_paged_attention_benchmark" \
        "${model_path}" >"${result_dir}/profile-program.txt"
    "${nsys_bin}" stats --report cuda_gpu_kern_sum --format csv \
        "${profile_base}.nsys-rep" \
        >"${result_dir}/kernel_cuda_gpu_kern_sum.csv"
    nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu \
        --format=csv,noheader >"${result_dir}/gpu-state-after.txt"
    rm -f "${profile_base}.nsys-rep" "${profile_base}.sqlite"
fi
echo "correctness=PASS" >>"${result_dir}/metadata.txt"
echo "cuda_paged_attention_ab_result_dir=${result_dir}"
echo "cuda_paged_attention_ab=PASS"
