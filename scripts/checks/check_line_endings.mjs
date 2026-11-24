#!/usr/bin/env node

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join } from "node:path";

const shellExtensions = new Set([".sh", ".bash"]);
const defaultRoots = ["code", "exercises", "scripts", "docker"];

function shellScriptsUnder(target) {
  if (!statSync(target).isDirectory()) return [target];

  const files = [];
  for (const entry of readdirSync(target, { withFileTypes: true })) {
    const path = join(target, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "build" || entry.name.startsWith("build-")) continue;
      files.push(...shellScriptsUnder(path));
    } else if (shellExtensions.has(extname(entry.name))) {
      files.push(path);
    }
  }
  return files;
}

const requestedRoots = process.argv.slice(2);
const roots = requestedRoots.length > 0
  ? requestedRoots
  : defaultRoots.filter(existsSync);
const targets = roots
  .flatMap(shellScriptsUnder)
  .sort();
const invalid = targets.filter((file) => readFileSync(file).includes(0x0d));

if (invalid.length > 0) {
  console.error("Shell script line ending check failed; CR bytes found:");
  for (const file of invalid) console.error(`- ${file}`);
  console.error("Save these files with LF line endings.");
  process.exit(1);
}

console.log(`Shell script line ending check passed: ${targets.length} files use LF`);
