#!/usr/bin/env node

// Chapter 08：比较同一真实 Decoder 可执行文件的 Q6_K down scalar/AVX2 结果。
// 契约：正确性、计时闭合和 workload 必须一致，down 与总时间都改善才保留实验。
// 验证：run_real_q6k_down_ab.sh 生成并归档三轮 A/B 报告。
import fs from "node:fs";

const [baselinePath, experimentPath] = process.argv.slice(2);
if (!baselinePath || !experimentPath) {
  console.error("usage: compare_q6k_down_profiles.mjs BASELINE EXPERIMENT");
  process.exit(2);
}

function parse(path) {
  const values = new Map();
  for (const line of fs.readFileSync(path, "utf8").split(/\r?\n/)) {
    const separator = line.indexOf("=");
    if (separator > 0) values.set(line.slice(0, separator), line.slice(separator + 1));
  }
  return values;
}

function number(values, name) {
  const value = Number(values.get(name));
  if (!Number.isFinite(value) || value < 0) throw new Error(`invalid ${name}`);
  return value;
}

function requirePass(values, name) {
  if (values.get(name) !== "PASS") throw new Error(`${name} did not pass`);
}

const baseline = parse(baselinePath);
const experiment = parse(experimentPath);
for (const values of [baseline, experiment]) {
  requirePass(values, "correctness");
  requirePass(values, "profile_complete");
  requirePass(values, "ffn_profile_complete");
}
if (baseline.get("decoder_profile") !== experiment.get("decoder_profile") ||
    baseline.get("iterations") !== experiment.get("iterations") ||
    baseline.get("forward_tokens_per_iteration") !==
      experiment.get("forward_tokens_per_iteration") ||
    baseline.get("ffn_down_backend") !== "scalar" ||
    experiment.get("ffn_down_backend") !== "avx2") {
  throw new Error("baseline and experiment are not comparable");
}

const baselineDown = number(baseline, "ffn_down_ms");
const experimentDown = number(experiment, "ffn_down_ms");
const baselineFfn = number(baseline, "ffn_ms");
const experimentFfn = number(experiment, "ffn_ms");
const baselineTotal = number(baseline, "total_ms");
const experimentTotal = number(experiment, "total_ms");
const delta = (before, after) => (after - before) * 100 / before;
const downDelta = delta(baselineDown, experimentDown);
const ffnDelta = delta(baselineFfn, experimentFfn);
const totalDelta = delta(baselineTotal, experimentTotal);
const accepted = downDelta < 0 && totalDelta < 0;

const report = [
  "experiment=tinyllama_q6k_down_single_variable",
  "correctness=PASS",
  `iterations=${baseline.get("iterations")}`,
  `forward_tokens_per_iteration=${baseline.get("forward_tokens_per_iteration")}`,
  "baseline_backend=scalar",
  "experiment_backend=avx2",
  `baseline_ffn_down_ms=${baselineDown.toFixed(4)}`,
  `experiment_ffn_down_ms=${experimentDown.toFixed(4)}`,
  `ffn_down_delta_percent=${downDelta.toFixed(4)}`,
  `baseline_ffn_ms=${baselineFfn.toFixed(4)}`,
  `experiment_ffn_ms=${experimentFfn.toFixed(4)}`,
  `ffn_delta_percent=${ffnDelta.toFixed(4)}`,
  `baseline_total_ms=${baselineTotal.toFixed(4)}`,
  `experiment_total_ms=${experimentTotal.toFixed(4)}`,
  `total_delta_percent=${totalDelta.toFixed(4)}`,
  `decision=${accepted ? "KEEP" : "REJECT"}`,
  `reason=${accepted ? "down and total latency improved" : "down or total latency did not improve"}`,
];
console.log(report.join("\n"));
process.exit(accepted ? 0 : 1);
