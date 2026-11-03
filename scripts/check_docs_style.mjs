#!/usr/bin/env node

import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

const roots = ['source/chapter00_quickstart', 'source/chapter01_tensor_loader',
  'source/chapter02_tensor_ops', 'source/chapter03_matmul', 'source/chapter04_attention',
  'source/chapter05_quantization', 'source/chapter06_generation', 'source/chapter07_benchmark',
  'source/chapter08_profiling', 'source/chapter09_serving', 'source/foundation',
  'source/guides', 'source/reference', 'docs/code', 'exercises/foundation']
const entryFiles = ['README.md', 'USAGE.md', 'docs/README.md', 'docs/index.md', 'docs/guide.md']
const chineseFirstTitleRoots = roots.slice(0, 5)
const patterns = [
  /##\s+这一页要解决什么/,
  /##\s+本节概览/,
  /这一页/,
  /本页/,
  /这一层/,
  /本层/,
  /收口/,
  /如果只保留一句话/,
  /当前工作重点/,
  /旧链接仍/,
  /正文页仍应/,
  /卡片写作规则/,
  /教程的维护成本/,
  /后续新增.*再补/
]

function walk(dir) {
  const files = []
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) files.push(...walk(path))
    else if (path.endsWith('.md')) files.push(path)
  }
  return files
}

const files = new Set(entryFiles)
roots.forEach((root) => walk(root).forEach((file) => files.add(file)))

const findings = []
for (const file of files) {
  const lines = readFileSync(file, 'utf8').split(/\r?\n/)
  lines.forEach((line, index) => {
    if (patterns.some((pattern) => pattern.test(line))) {
      findings.push(`${file}:${index + 1}: ${line.trim()}`)
    }
    if (chineseFirstTitleRoots.some((root) => file.startsWith(`${root}/`)) &&
        /^#\s+\d+\.\d+\s+[A-Za-z]/.test(line)) {
      findings.push(`${file}:${index + 1}: chapter title must put the Chinese concept first`)
    }
  })
}

if (findings.length > 0) {
  console.error('Documentation style check failed:')
  findings.forEach((finding) => console.error(`- ${finding}`))
  process.exit(1)
} else {
  console.log(`Documentation style check passed: ${files.size} learner-facing files checked.`)
}
