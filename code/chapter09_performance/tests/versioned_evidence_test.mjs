// Chapter 09：验证仓库内真实 benchmark 原始样本与统一证据记录保持一致。
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { migrateLegacyDocument } from "../scripts/import_legacy_evidence.mjs";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const resultsDir = path.resolve(testDir, "../results");
const cases = [
  ["scalar", "sequential"],
  ["scalar", "parallel-requests"],
  ["optimized-avx2", "sequential"],
  ["optimized-avx2", "parallel-requests"],
];

for (const [policy, dispatch] of cases) {
  const rawPath = path.join(
    resultsDir,
    "raw",
    `tinyllama-service-${policy}-${dispatch}.json`,
  );
  const evidencePath = path.join(
    resultsDir,
    "evidence",
    `tinyllama_service_core_cpu-${policy}-${dispatch}.json`,
  );
  const raw = JSON.parse(fs.readFileSync(rawPath, "utf8"));
  const evidence = JSON.parse(fs.readFileSync(evidencePath, "utf8"));
  const records = migrateLegacyDocument(raw);

  assert.equal(records.length, 1, `${policy}/${dispatch}: record count`);
  assert.deepEqual(evidence, records[0], `${policy}/${dispatch}: evidence drift`);
  assert.equal(raw.iterations, raw.samples.length, `${policy}/${dispatch}: raw iterations`);
  assert.equal(raw.correctness.all_iterations, true, `${policy}/${dispatch}: raw correctness`);
  assert.ok(raw.samples.every((sample) => sample.correctness === true));
  assert.equal(evidence.correctness_pass, true);
  assert.equal(evidence.workload.measurement_iterations, evidence.samples.length);
  assert.deepEqual(evidence.workload.prompt_tokens, [6, 1]);
  assert.equal(evidence.workload.output_tokens_per_request, 4);
  assert.ok(evidence.samples.every((sample) =>
    sample.total_ms > 0 && sample.tokens_per_second > 0));
}

const profileText = fs.readFileSync(
  path.join(resultsDir, "profile", "tinyllama-decoder-scalar.txt"),
  "utf8",
);
const profile = Object.fromEntries(profileText.trim().split("\n").map((line) => {
  const separator = line.indexOf("=");
  return [line.slice(0, separator), line.slice(separator + 1)];
}));
assert.equal(profile.experiment_id, "EXP-CPU-04");
assert.equal(profile.workload_id, "WL-CPU-TINYLLAMA-DECODER-PHASE-V1");
assert.equal(profile.model, "TinyLlama-1.1B-Chat-v1.0.Q4_K_M");
assert.ok(profile.cpu_model.length > 0);
assert.equal(profile.build_type, "Release");
assert.equal(profile.correctness, "PASS");
assert.equal(profile.profile_complete, "PASS");
assert.equal(profile.iterations, "3");

const baseline = JSON.parse(fs.readFileSync(path.join(
  resultsDir,
  "evidence",
  "tinyllama_service_core_cpu-optimized-avx2-sequential.json",
), "utf8"));
const experiment = JSON.parse(fs.readFileSync(path.join(
  resultsDir,
  "evidence",
  "tinyllama_service_core_cpu-optimized-avx2-parallel-requests.json",
), "utf8"));
const decision = JSON.parse(fs.readFileSync(path.join(
  resultsDir,
  "evidence",
  "tinyllama_service_core_cpu-avx2-dispatch-decision.json",
), "utf8"));
assert.equal(decision.baseline_experiment_id, baseline.experiment_id);
assert.equal(decision.experiment_experiment_id, experiment.experiment_id);
assert.equal(decision.workload_id, baseline.workload_id);
assert.equal(decision.policy.primary_metric, "throughput");
assert.equal(decision.decision, "KEEP");

process.stdout.write("versioned performance evidence tests passed\n");
