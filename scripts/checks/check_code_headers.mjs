import fs from "node:fs";
import path from "node:path";

const sourceExtensions = new Set([".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh"]);
const roots = process.argv.slice(2);

if (roots.length === 0) {
  console.error("Usage: node scripts/checks/check_code_headers.mjs <directory> [...]");
  process.exit(2);
}

function filesUnder(root) {
  const result = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "build" || entry.name.startsWith("build-")) continue;
      result.push(...filesUnder(fullPath));
    } else if (sourceExtensions.has(path.extname(entry.name))) {
      result.push(fullPath);
    }
  }
  return result;
}

const missing = [];
for (const root of roots) {
  if (!fs.existsSync(root)) {
    missing.push(`${root} (directory does not exist)`);
    continue;
  }
  for (const file of filesUnder(root)) {
    const firstLine = fs.readFileSync(file, "utf8").split(/\r?\n/, 1)[0].trim();
    if (!firstLine.startsWith("//")) missing.push(file);
  }
}

if (missing.length > 0) {
  console.error("Missing source file headers:");
  for (const file of missing) console.error(`- ${file}`);
  process.exit(1);
}

console.log(`Code header check passed: ${roots.join(", ")}`);
