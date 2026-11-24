#!/usr/bin/env node

import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const catalogPath = 'scripts/README.md'
const startMarker = '<!-- script-catalog:start -->'
const endMarker = '<!-- script-catalog:end -->'
const allowedLevels = new Set(['L0', 'L1', 'L2', 'L3', '维护工具'])
const scriptPattern = /\.(?:mjs|py|sh)$/

const catalog = readFileSync(catalogPath, 'utf8')
const start = catalog.indexOf(startMarker)
const end = catalog.indexOf(endMarker)
if (start < 0 || end <= start) {
  console.error(`${catalogPath}: script catalog markers are missing or out of order`)
  process.exit(1)
}

const entries = new Map()
const rowPattern = /^\| `([^`]+)` \| ([^|]+?) \|/gm
const catalogBody = catalog.slice(start + startMarker.length, end)
for (const match of catalogBody.matchAll(rowPattern)) {
  const [, name, levelText] = match
  const level = levelText.trim()
  if (entries.has(name)) {
    console.error(`${catalogPath}: duplicate script entry: ${name}`)
    process.exit(1)
  }
  if (!allowedLevels.has(level)) {
    console.error(`${catalogPath}: unsupported level '${level}' for ${name}`)
    process.exit(1)
  }
  entries.set(name, level)
}

function collectScripts(dir, prefix = '') {
  const scripts = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === 'local') continue
    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name
    if (entry.isDirectory()) {
      scripts.push(...collectScripts(join(dir, entry.name), relativePath))
    } else if (scriptPattern.test(entry.name)) {
      scripts.push(relativePath)
    }
  }
  return scripts
}

const scripts = collectScripts('scripts').sort()
const registered = [...entries.keys()].sort()
const missing = scripts.filter((name) => !entries.has(name))
const stale = registered.filter((name) => !scripts.includes(name))

if (missing.length > 0 || stale.length > 0) {
  console.error('Script catalog check failed:')
  for (const name of missing) console.error(`- scripts/${name}: missing from ${catalogPath}`)
  for (const name of stale) console.error(`- ${catalogPath}: scripts/${name} does not exist`)
  process.exit(1)
}

console.log(`Script catalog check passed: ${scripts.length} scripts across L0-L3 and maintenance.`)
