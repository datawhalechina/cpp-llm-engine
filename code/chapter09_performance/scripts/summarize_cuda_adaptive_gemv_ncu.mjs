// Chapter 09：从 NCU CSV 提取自适应 GEMV 的原因指标。
// 原始 CSV 保持不变；摘要只选择跨当前实验有解释力的计数器。
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SELECTED_METRICS = [
  "Elapsed Cycles",
  "Duration",
  "Registers Per Thread",
  "Static Shared Memory Per Block",
  "Achieved Occupancy",
  "Executed Instructions",
  "Issued Instructions",
  "Executed Ipc Active",
  "Active Warps Per Scheduler",
  "Eligible Warps Per Scheduler",
  "Issued Warp Per Scheduler",
  "Warp Cycles Per Issued Instruction",
  "Warp Cycles Per Executed Instruction",
  "Avg. Active Threads Per Warp",
  "Avg. Not Predicated Off Threads Per Warp",
  "Memory Throughput",
  "Mem Busy",
  "Mem Pipes Busy",
  "L1/TEX Hit Rate",
  "L2 Hit Rate",
  "DRAM Throughput",
  "L1/TEX Cache Throughput",
  "L2 Cache Throughput",
];

function parseCsvLine(line) {
  const result = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; ++index) {
    const character = line[index];
    if (character === '"' && line[index + 1] === '"') {
      value += '"';
      ++index;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      result.push(value);
      value = "";
    } else {
      value += character;
    }
  }
  result.push(value);
  return result;
}

function parseNcu(file) {
  const lines = fs.readFileSync(file, "utf8").split("\n");
  const headerIndex = lines.findIndex((line) => line.startsWith('"ID",'));
  if (headerIndex < 0) throw new Error(`${path.basename(file)} has no NCU CSV header`);
  const header = parseCsvLine(lines[headerIndex]);
  const metricName = header.indexOf("Metric Name");
  const metricUnit = header.indexOf("Metric Unit");
  const metricValue = header.indexOf("Metric Value");
  const metrics = {};
  for (const line of lines.slice(headerIndex + 1)) {
    if (!line.startsWith('"')) continue;
    const row = parseCsvLine(line);
    if (!SELECTED_METRICS.includes(row[metricName])) continue;
    const numeric = Number(row[metricValue].replaceAll(",", ""));
    if (Number.isFinite(numeric)) {
      metrics[row[metricName]] = { value: numeric, unit: row[metricUnit] };
    }
  }
  return metrics;
}

function deltaPercent(baseline, experiment) {
  if (baseline === 0) return null;
  return (experiment - baseline) / baseline * 100;
}

export function summarizeAdaptiveGemvNcu(resultDir) {
  const comparisons = [];
  for (const format of ["q4", "q6"]) {
    for (const batch of [1, 2]) {
      const fixed = parseNcu(path.join(resultDir, `ncu-${format}-b${batch}-fixed-four.csv`));
      const adaptive = parseNcu(path.join(resultDir, `ncu-${format}-b${batch}-adaptive.csv`));
      const deltas = {};
      for (const metric of SELECTED_METRICS) {
        if (fixed[metric] && adaptive[metric]) {
          deltas[metric] = {
            baseline: fixed[metric].value,
            experiment: adaptive[metric].value,
            unit: adaptive[metric].unit,
            delta_percent: deltaPercent(fixed[metric].value, adaptive[metric].value),
          };
        }
      }
      comparisons.push({ format: format.toUpperCase(), batch, metrics: deltas });
    }
  }
  return { schema_version: 1, profiler: "Nsight Compute", comparisons };
}

function main() {
  const [resultDir, outputFile] = process.argv.slice(2);
  if (!resultDir || !outputFile) {
    console.error("usage: summarize_cuda_adaptive_gemv_ncu.mjs <result-dir> <output.json>");
    process.exit(2);
  }
  try {
    fs.writeFileSync(
      outputFile,
      `${JSON.stringify(summarizeAdaptiveGemvNcu(resultDir), null, 2)}\n`,
    );
    process.stdout.write(`cuda_adaptive_gemv_ncu_summary=${outputFile}\n`);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 2;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) main();
