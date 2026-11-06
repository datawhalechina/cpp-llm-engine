#!/usr/bin/env node
// 检查术语和公式参考资产的稳定编号、引用与重复项。
import fs from 'node:fs'
import path from 'node:path'

const formulaFile = 'source/reference/formulas.md'
const glossaryFile = 'source/reference/glossary.md'
const chapterAssetsFile = 'source/reference/chapter_assets.md'
const experimentAssetsFile = 'source/reference/experiment_assets.md'
const sourceRoot = 'source'

const formulaText = fs.readFileSync(formulaFile, 'utf8')
const glossaryText = fs.readFileSync(glossaryFile, 'utf8')
const chapterAssetsText = fs.readFileSync(chapterAssetsFile, 'utf8')
const experimentAssetsText = fs.readFileSync(experimentAssetsFile, 'utf8')
const findings = []

function markdownFiles(root) {
  const files = []
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const fullPath = path.join(root, entry.name)
    if (entry.isDirectory()) files.push(...markdownFiles(fullPath))
    if (entry.isFile() && entry.name.endsWith('.md')) files.push(fullPath)
  }
  return files
}

const formulaIds = []
for (const match of formulaText.matchAll(/<a id="(f-[a-z0-9-]+)"><\/a>\s*\n###\s+(F-[A-Z0-9-]+)/g)) {
  formulaIds.push(match[1])
  if (match[1].toUpperCase() !== match[2]) {
    findings.push(`${formulaFile}: anchor ${match[1]} does not match heading ${match[2]}`)
  }
}

const formulaIdSet = new Set(formulaIds)
if (formulaIds.length !== formulaIdSet.size) {
  findings.push(`${formulaFile}: duplicate formula IDs found`)
}

const requiredFormulaIds = [
  'f-tensor-01',
  'f-load-01',
  'f-linear-01',
  'f-norm-02',
  'f-attn-02',
  'f-rope-01',
  'f-ffn-01',
  'f-kv-01',
  'f-sched-01',
  'f-quant-01',
  'f-cuda-01',
  'f-sample-01',
  'f-perf-01',
  'f-serve-01',
  'f-mm-01',
  'f-mm-03',
]
for (const id of requiredFormulaIds) {
  if (!formulaIdSet.has(id)) findings.push(`${formulaFile}: missing required formula ${id}`)
}

for (const file of markdownFiles(sourceRoot)) {
  const text = fs.readFileSync(file, 'utf8')
  for (const match of text.matchAll(/formulas\.md#(f-[a-z0-9-]+)/g)) {
    if (!formulaIdSet.has(match[1])) {
      findings.push(`${file}: references unknown formula ${match[1]}`)
    }
  }
}

const terms = []
for (const line of glossaryText.split(/\r?\n/)) {
  if (!line.startsWith('|') || /^\|\s*(?:---|中文术语|名称)/.test(line)) continue
  const firstCell = line.split('|')[1]?.trim()
  if (firstCell) terms.push(firstCell)
}
const termSet = new Set(terms)
if (terms.length !== termSet.size) {
  const duplicates = [...termSet].filter((term) => terms.indexOf(term) !== terms.lastIndexOf(term))
  findings.push(`${glossaryFile}: duplicate terms: ${duplicates.join(', ')}`)
}

const requiredTerms = [
  '张量',
  '后端',
  '分词器',
  '模型元数据',
  '注意力',
  '键值缓存',
  '词元预算',
  '首词元延迟',
  '准入控制',
  '多模态推理',
  '基准结果',
]
for (const term of requiredTerms) {
  if (!termSet.has(term)) findings.push(`${glossaryFile}: missing required term ${term}`)
}

const chapterSlugs = [
  'chapter00_quickstart',
  'chapter01_tensor_loader',
  'chapter02_tensor_ops',
  'chapter03_matmul',
  'chapter04_attention',
  'chapter06_generation',
  'chapter05_quantization',
  'chapter07_kv_cache',
  'chapter08_scheduling',
  'chapter09_performance',
  'chapter10_serving',
]

const restructuredChapterAssets = {
  chapter06_generation: ['../chapter06_generation/next/intro.md', '/diagrams/chapter06-generation-core.svg'],
  chapter07_kv_cache: ['../chapter07_kv_cache/next/intro.md', '/diagrams/chapter07-kv-cache.svg'],
  chapter08_scheduling: ['../chapter08_scheduling/next/intro.md', '/diagrams/chapter08-scheduling.svg'],
  chapter09_performance: ['../chapter09_performance/next/intro.md', '/diagrams/chapter09-performance.svg'],
  chapter10_serving: ['../chapter10_serving/next/intro.md', '/diagrams/chapter10-serving.svg'],
}

for (const slug of chapterSlugs) {
  const chapterNumber = slug.slice(7, 9)
  const [introAsset, diagramAsset] = restructuredChapterAssets[slug] ?? [
    `../${slug}/intro.md`,
    `/diagrams/chapter${chapterNumber}-main.svg`,
  ]
  const requiredAssets = [
    introAsset,
    `/code/${slug}`,
    diagramAsset,
  ]
  for (const asset of requiredAssets) {
    if (!chapterAssetsText.includes(asset)) {
      findings.push(`${chapterAssetsFile}: missing ${slug} asset ${asset}`)
    }
  }

  const codeReadme = `code/${slug}/README.md`
  const codeReadmeText = fs.readFileSync(codeReadme, 'utf8')
  if (!/^## (?:本章)?资产总览$/m.test(codeReadmeText)) {
    findings.push(`${codeReadme}: missing asset overview heading`)
  }
}

const experimentIds = [
  ...experimentAssetsText.matchAll(/<a id="((?:exp|val)-[a-z0-9-]+)"><\/a>/g),
].map((match) => match[1])
const experimentIdSet = new Set(experimentIds)
if (experimentIds.length !== experimentIdSet.size) {
  findings.push(`${experimentAssetsFile}: duplicate experiment IDs found`)
}

const requiredExperimentIds = [
  'exp-cpu-01',
  'exp-cpu-02',
  'exp-cpu-04',
  'exp-cpu-08',
  'val-mm-01',
  'val-mm-05',
  'val-mm-08',
  'exp-mm-01',
  'exp-gpu-01',
]
for (const id of requiredExperimentIds) {
  if (!experimentIdSet.has(id)) {
    findings.push(`${experimentAssetsFile}: missing required experiment ${id}`)
  }
}

for (const match of experimentAssetsText.matchAll(/href="\.\.\/_code\/([^"]+)"/g)) {
  const asset = `code/${match[1]}`
  if (!fs.existsSync(asset)) {
    findings.push(`${experimentAssetsFile}: experiment asset does not exist: ${asset}`)
  }
}

if (findings.length > 0) {
  console.error('Reference asset check failed:')
  for (const finding of findings) console.error(`- ${finding}`)
  process.exit(1)
}

console.log(
  `Reference asset check passed: ${formulaIds.length} formulas, ${terms.length} terms, ` +
    `${chapterSlugs.length} chapters, ${experimentIds.length} experiments`,
)

