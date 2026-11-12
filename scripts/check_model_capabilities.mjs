#!/usr/bin/env node
// Ensure every declared model capability has a valid state and inspectable evidence.
import fs from 'node:fs'
import path from 'node:path'

const repoRoot = process.cwd()
const manifestPath = path.join(repoRoot, 'code/testing/model_capabilities.json')
const docsPath = path.join(repoRoot, 'source/reference/model_support.md')
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
const modelSupport = fs.readFileSync(docsPath, 'utf8')
const findings = []
const modelIds = new Set()
const states = new Set(manifest.allowed_states ?? [])
const capabilities = manifest.capability_order ?? []

if (manifest.schema_version !== 1) findings.push('unsupported schema_version')
if (states.size === 0) findings.push('allowed_states must be non-empty')
if (capabilities.length === 0) findings.push('capability_order must be non-empty')

for (const model of manifest.models ?? []) {
  if (!model.id || !model.name || !model.family) {
    findings.push(`${model.id ?? '<unknown>'}: id, name, and family are required`)
    continue
  }
  if (modelIds.has(model.id)) findings.push(`${model.id}: duplicate model id`)
  modelIds.add(model.id)
  if (!modelSupport.includes(model.name)) {
    findings.push(`${model.id}: name is missing from source/reference/model_support.md`)
  }

  for (const capability of capabilities) {
    const entry = model.capabilities?.[capability]
    if (!entry) {
      findings.push(`${model.id}: missing capability '${capability}'`)
      continue
    }
    if (!states.has(entry.state)) {
      findings.push(`${model.id}/${capability}: unknown state '${entry.state}'`)
    }
    if (!Array.isArray(entry.evidence)) {
      findings.push(`${model.id}/${capability}: evidence must be an array`)
      continue
    }
    if (['verified', 'partial'].includes(entry.state) && entry.evidence.length === 0) {
      findings.push(`${model.id}/${capability}: ${entry.state} requires evidence`)
    }
    if (['partial', 'planned'].includes(entry.state) && !entry.gap) {
      findings.push(`${model.id}/${capability}: ${entry.state} requires a gap description`)
    }
    for (const evidence of entry.evidence) {
      if (path.isAbsolute(evidence) || evidence.includes('..')) {
        findings.push(`${model.id}/${capability}: invalid evidence path '${evidence}'`)
      } else if (!fs.existsSync(path.join(repoRoot, evidence))) {
        findings.push(`${model.id}/${capability}: evidence does not exist '${evidence}'`)
      }
    }
  }

  for (const capability of Object.keys(model.capabilities ?? {})) {
    if (!capabilities.includes(capability)) {
      findings.push(`${model.id}: undeclared capability '${capability}'`)
    }
  }
}

if (findings.length > 0) {
  console.error('Model capability check failed:')
  for (const finding of findings) console.error(`- ${finding}`)
  process.exit(1)
}

console.log(
  `Model capability check passed: ${modelIds.size} models, ` +
  `${capabilities.length} capability stages`,
)
