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

process.stdout.write("versioned performance evidence tests passed\n");
