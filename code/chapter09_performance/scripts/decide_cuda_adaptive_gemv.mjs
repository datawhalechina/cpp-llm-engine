// Chapter 09：根据两轮真实 Decode A/B 样本给出量化 GEMV 路径决策。
// 契约：正确性、批量收益和顺序路径保护指标必须分别通过。
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const METRICS = [
  "sequential_decode_median_ms",
  "tensor_batch_decode_median_ms",
  "logits_max_error",
];

function parseOutput(file) {
  const fields = Object.fromEntries(fs.readFileSync(file, "utf8")
    .trim().split("\n").map((line) => {
      const separator = line.indexOf(":");
      return [line.slice(0, separator).trim(), line.slice(separator + 1).trim()];
    }));
  for (const metric of METRICS) {
    if (!Number.isFinite(Number(fields[metric]))) {
      throw new Error(`${path.basename(file)} missing numeric ${metric}`);
    }
  }
  return Object.fromEntries(METRICS.map((metric) => [metric, Number(fields[metric])]));
}

function deltaPercent(baseline, experiment) {
  return (experiment - baseline) / baseline * 100;
}

export function decideAdaptiveGemv(resultDir, policy = {}) {
  const minimumBatchImprovementPercent = policy.minimumBatchImprovementPercent ?? 5;
  const maximumSequentialRegressionPercent = policy.maximumSequentialRegressionPercent ?? 5;
  const maximumLogitsError = policy.maximumLogitsError ?? 1e-3;
  const runs = [1, 2].map((run) => {
    const fixed = parseOutput(path.join(resultDir, `fixed-four-run-${run}.txt`));
    const adaptive = parseOutput(path.join(resultDir, `adaptive-run-${run}.txt`));
    return {
      run,
      fixed,
      adaptive,
      sequential_delta_percent: deltaPercent(
        fixed.sequential_decode_median_ms,
        adaptive.sequential_decode_median_ms),
      batch_delta_percent: deltaPercent(
        fixed.tensor_batch_decode_median_ms,
        adaptive.tensor_batch_decode_median_ms),
    };
  });
  const correctnessPass = runs.every(({ fixed, adaptive }) =>
    fixed.logits_max_error <= maximumLogitsError &&
    adaptive.logits_max_error <= maximumLogitsError);
  const sequentialGuardPass = runs.every(({ sequential_delta_percent: delta }) =>
    delta <= maximumSequentialRegressionPercent);
  const batchImprovementPass = runs.every(({ batch_delta_percent: delta }) =>
    delta <= -minimumBatchImprovementPercent);

  let decision = "INCONCLUSIVE";
  let reason = "batch improvement is not stable across independent runs";
  if (!correctnessPass) {
    decision = "REJECT";
    reason = "logits correctness guard failed";
  } else if (!sequentialGuardPass) {
    decision = "REJECT";
    reason = "sequential Decode regression exceeded the guard";
  } else if (batchImprovementPass) {
    decision = "KEEP";
    reason = "both runs improved Tensor Batch and preserved sequential Decode";
  }
  return {
    schema_version: 1,
    execution_status: "PASS",
    correctness: correctnessPass ? "PASS" : "FAIL",
    decision,
    reason,
    policy: {
      minimum_batch_improvement_percent: minimumBatchImprovementPercent,
      maximum_sequential_regression_percent: maximumSequentialRegressionPercent,
      maximum_logits_error: maximumLogitsError,
    },
    runs,
  };
}

function main() {
  const [resultDir, outputFile] = process.argv.slice(2);
  if (!resultDir || !outputFile) {
    console.error("usage: decide_cuda_adaptive_gemv.mjs <result-dir> <output.json>");
    process.exit(2);
  }
  try {
    const result = decideAdaptiveGemv(resultDir);
    fs.writeFileSync(outputFile, `${JSON.stringify(result, null, 2)}\n`);
    process.stdout.write(`cuda_adaptive_gemv_decision=${result.decision}\n`);
    if (result.decision !== "KEEP") process.exitCode = 1;
  } catch (error) {
    console.error(error.message);
    process.exitCode = 2;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) main();
