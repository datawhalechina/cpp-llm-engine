// Chapter 09：合并两轮真实 Adaptive GEMV 运行矩阵并计算逐单元变化率。
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

function parseMatrix(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split("\n");
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(header.map((name, index) => {
      const value = values[index];
      return [name, name === "path" ? value : Number(value)];
    }));
  });
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function deltaPercent(baseline, experiment) {
  return (experiment - baseline) / baseline * 100;
}

export function summarizeAdaptiveGemvMatrix(resultDir) {
  const rows = [1, 2].flatMap((run) =>
    parseMatrix(path.join(resultDir, `run-${run}.csv`)).map((row) => ({ ...row, run })));
  const keys = [...new Set(rows.map((row) =>
    `${row.batch}/${row.context_tokens}/${row.decode_steps}`))];
  const cells = keys.map((key) => {
    const [batch, contextTokens, decodeSteps] = key.split("/").map(Number);
    const selected = rows.filter((row) =>
      row.batch === batch && row.context_tokens === contextTokens &&
      row.decode_steps === decodeSteps);
    const fixed = selected.filter((row) => row.path === "fixed-four");
    const adaptive = selected.filter((row) => row.path === "adaptive");
    if (fixed.length !== 2 || adaptive.length !== 2) {
      throw new Error(`matrix cell ${key} does not contain two runs per path`);
    }
    const fixedBatch = median(fixed.map((row) => row.tensor_batch_median_ms));
    const adaptiveBatch = median(adaptive.map((row) => row.tensor_batch_median_ms));
    const fixedSequential = median(fixed.map((row) => row.sequential_median_ms));
    const adaptiveSequential = median(adaptive.map((row) => row.sequential_median_ms));
    const maximumError = Math.max(...selected.map((row) => row.logits_max_error));
    return {
      batch,
      context_tokens: contextTokens,
      decode_steps: decodeSteps,
      fixed_batch_median_ms: fixedBatch,
      adaptive_batch_median_ms: adaptiveBatch,
      batch_delta_percent: deltaPercent(fixedBatch, adaptiveBatch),
      fixed_sequential_median_ms: fixedSequential,
      adaptive_sequential_median_ms: adaptiveSequential,
      sequential_delta_percent: deltaPercent(fixedSequential, adaptiveSequential),
      logits_max_error: maximumError,
    };
  });
  const correctness = cells.every((cell) => cell.logits_max_error <= 1e-3);
  return {
    schema_version: 1,
    execution_status: "PASS",
    correctness: correctness ? "PASS" : "FAIL",
    cells,
  };
}

function main() {
  const [resultDir, outputFile] = process.argv.slice(2);
  if (!resultDir || !outputFile) {
    console.error("usage: summarize_cuda_adaptive_gemv_matrix.mjs <result-dir> <output.json>");
    process.exit(2);
  }
  try {
    const result = summarizeAdaptiveGemvMatrix(resultDir);
    fs.writeFileSync(outputFile, `${JSON.stringify(result, null, 2)}\n`);
    process.stdout.write(`cuda_adaptive_gemv_matrix_correctness=${result.correctness}\n`);
    if (result.correctness !== "PASS") process.exitCode = 1;
  } catch (error) {
    console.error(error.message);
    process.exitCode = 2;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) main();
