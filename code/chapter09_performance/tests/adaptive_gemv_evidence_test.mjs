// Chapter 09：验证 CUDA Adaptive GEMV 的自动决策和 NCU 摘要。
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { decideAdaptiveGemv } from "../scripts/decide_cuda_adaptive_gemv.mjs";
import { summarizeAdaptiveGemvNcu } from "../scripts/summarize_cuda_adaptive_gemv_ncu.mjs";
import { summarizeAdaptiveGemvMatrix } from "../scripts/summarize_cuda_adaptive_gemv_matrix.mjs";

const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "chapter09-adaptive-gemv-"));

function writeRuntime(name, sequential, batch, error = 0) {
  fs.writeFileSync(path.join(temporary, name), [
    `sequential_decode_median_ms: ${sequential}`,
    `tensor_batch_decode_median_ms: ${batch}`,
    `logits_max_error: ${error}`,
  ].join("\n"));
}

function writeNcu(name, cycles, shared) {
  const header = '"ID","Metric Name","Metric Unit","Metric Value"';
  fs.writeFileSync(path.join(temporary, name), [
    header,
    `"0","Elapsed Cycles","cycle","${cycles}"`,
    `"0","Executed Instructions","inst","${cycles * 10}"`,
    `"0","Static Shared Memory Per Block","byte/block","${shared}"`,
  ].join("\n"));
}

try {
  for (const run of [1, 2]) {
    writeRuntime(`fixed-four-run-${run}.txt`, 30, 16);
    writeRuntime(`adaptive-run-${run}.txt`, 30.5, 13);
  }
  const keep = decideAdaptiveGemv(temporary);
  assert.equal(keep.execution_status, "PASS");
  assert.equal(keep.correctness, "PASS");
  assert.equal(keep.decision, "KEEP");
  assert.ok(keep.runs.every((run) => run.batch_delta_percent < -15));

  writeRuntime("adaptive-run-2.txt", 32, 13);
  assert.equal(decideAdaptiveGemv(temporary).decision, "REJECT");
  writeRuntime("adaptive-run-2.txt", 30.5, 15.5);
  assert.equal(decideAdaptiveGemv(temporary).decision, "INCONCLUSIVE");
  writeRuntime("adaptive-run-2.txt", 30.5, 13, 0.1);
  assert.equal(decideAdaptiveGemv(temporary).decision, "REJECT");

  for (const format of ["q4", "q6"]) {
    for (const batch of [1, 2]) {
      writeNcu(`ncu-${format}-b${batch}-fixed-four.csv`, 100000, 128);
      writeNcu(`ncu-${format}-b${batch}-adaptive.csv`, 80000, batch * 32);
    }
  }
  const summary = summarizeAdaptiveGemvNcu(temporary);
  assert.equal(summary.comparisons.length, 4);
  assert.equal(
    summary.comparisons[0].metrics["Elapsed Cycles"].delta_percent,
    -20,
  );
  assert.equal(
    summary.comparisons[0].metrics["Static Shared Memory Per Block"].experiment,
    32,
  );
  assert.equal(
    summary.comparisons[0].metrics["Executed Instructions"].delta_percent,
    -20,
  );

  const matrixHeader = "path,batch,context_tokens,decode_steps,sequential_median_ms,tensor_batch_median_ms,batch_over_sequential,logits_max_error";
  fs.writeFileSync(path.join(temporary, "run-1.csv"), [
    matrixHeader,
    "fixed-four,2,16,8,30,16,0.53,0",
    "adaptive,2,16,8,30.5,13,0.43,0",
  ].join("\n"));
  fs.writeFileSync(path.join(temporary, "run-2.csv"), [
    matrixHeader,
    "adaptive,2,16,8,29.5,12.5,0.42,0",
    "fixed-four,2,16,8,30,15.5,0.52,0",
  ].join("\n"));
  const matrix = summarizeAdaptiveGemvMatrix(temporary);
  assert.equal(matrix.correctness, "PASS");
  assert.equal(matrix.cells.length, 1);
  assert.ok(matrix.cells[0].batch_delta_percent < -15);
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}

process.stdout.write("adaptive GEMV evidence tests passed\n");
