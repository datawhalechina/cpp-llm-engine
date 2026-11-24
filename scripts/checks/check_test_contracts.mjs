#!/usr/bin/env node
// 检查章节测试的契约所有者、依赖方向、证据层级和迁移债务。
import fs from 'node:fs'
import path from 'node:path'

const repoRoot = process.cwd()
const manifestPath = path.join(repoRoot, 'code/testing/chapter_test_contracts.json')
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
const findings = []
const trackedSources = new Map()
const trackedTargets = new Set()
const levels = new Set(manifest.allowed_levels)

if (!Array.isArray(manifest.audited_modules) || manifest.audited_modules.length === 0) {
  findings.push('manifest: audited_modules must be a non-empty array')
}

function chapterFromSource(source) {
  const match = source.match(/code\/chapter(\d{2})_[^/]+\/tests\//)
  return match?.[1]
}

function moduleRootFromSource(source) {
  for (const marker of ['/tests/', '/examples/']) {
    const index = source.indexOf(marker)
    if (index >= 0) return source.slice(0, index)
  }
  return undefined
}

function filesUnder(root) {
  if (!fs.existsSync(root)) return []
  const files = []
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const fullPath = path.join(root, entry.name)
    if (entry.isDirectory()) files.push(...filesUnder(fullPath))
    if (entry.isFile() && /(?:test|tests)\.(?:cpp|cu)$/.test(entry.name)) {
      files.push(path.relative(repoRoot, fullPath).split(path.sep).join('/'))
    }
  }
  return files
}

for (const test of manifest.tests) {
  for (const field of ['target', 'source', 'owner', 'level', 'dependencies', 'status']) {
    if (test[field] === undefined) findings.push(`${test.target ?? test.source}: misses ${field}`)
  }
  if (trackedTargets.has(test.target)) findings.push(`${test.target}: duplicate target`)
  trackedTargets.add(test.target)
  if (trackedSources.has(test.source)) findings.push(`${test.source}: duplicate source`)
  trackedSources.set(test.source, test)

  const additionalSources = test.additional_sources ?? []
  if (!Array.isArray(additionalSources)) {
    findings.push(`${test.target}: additional_sources must be an array`)
  } else {
    for (const source of additionalSources) {
      if (trackedSources.has(source)) findings.push(`${source}: duplicate source`)
      trackedSources.set(source, test)
      if (!fs.existsSync(path.join(repoRoot, source))) {
        findings.push(`${source}: source does not exist`)
      }
    }
  }

  const absoluteSource = path.join(repoRoot, test.source)
  if (!fs.existsSync(absoluteSource)) {
    findings.push(`${test.source}: source does not exist`)
    continue
  }
  if (!levels.has(test.level)) findings.push(`${test.target}: unknown level '${test.level}'`)
  if (!['active', 'legacy'].includes(test.status)) {
    findings.push(`${test.target}: unknown status '${test.status}'`)
  }
  const locationChapter = chapterFromSource(test.source)
  if (test.status === 'active' && locationChapter && locationChapter !== test.owner) {
    findings.push(
      `${test.target}: active owner Chapter${test.owner} does not match ` +
      `source location Chapter${locationChapter}`,
    )
  }
  if (test.status === 'legacy' && !test.reason) {
    findings.push(`${test.target}: legacy entry requires a migration reason`)
  }
  for (const dependency of test.dependencies) {
    if (!/^\d{2}$/.test(dependency)) {
      findings.push(`${test.target}: invalid dependency '${dependency}'`)
      continue
    }
    if (Number(dependency) > Number(test.owner)) {
      findings.push(
        `${test.target}: owner Chapter${test.owner} depends on future Chapter${dependency}`,
      )
    }
  }

  const moduleRoot = moduleRootFromSource(test.source)
  if (!moduleRoot) {
    findings.push(`${test.source}: source must be under tests/ or examples/`)
    continue
  }
  const cmakePath = path.join(repoRoot, moduleRoot, 'CMakeLists.txt')
  const cmake = fs.readFileSync(cmakePath, 'utf8')
  const relativeSource = test.source.slice(moduleRoot.length + 1)
  if (!cmake.includes(test.target)) findings.push(`${test.target}: target missing from ${moduleRoot}/CMakeLists.txt`)
  if (!cmake.includes(relativeSource)) findings.push(`${test.source}: source missing from ${moduleRoot}/CMakeLists.txt`)
  for (const source of additionalSources) {
    const relativeAdditionalSource = source.slice(moduleRoot.length + 1)
    if (!cmake.includes(relativeAdditionalSource)) {
      findings.push(`${source}: source missing from ${moduleRoot}/CMakeLists.txt`)
    }
  }
  const contractCmakePath = test.contract_cmake
    ? path.join(repoRoot, test.contract_cmake)
    : cmakePath
  if (!fs.existsSync(contractCmakePath)) {
    findings.push(`${test.target}: contract CMake file does not exist`)
  } else if (test.status === 'active') {
    const contractCmake = fs.readFileSync(contractCmakePath, 'utf8')
    if (!contractCmake.includes(test.target) ||
        !contractCmake.includes('cpp_llm_set_test_contract')) {
      findings.push(`${test.target}: ownership contract missing from ${path.relative(repoRoot, contractCmakePath)}`)
    }
  }
}

for (const moduleName of manifest.audited_modules ?? []) {
  if (!/^chapter\d{2}_[a-z0-9_]+$/.test(moduleName)) {
    findings.push(`${moduleName}: invalid audited module name`)
    continue
  }
  const modulePath = path.join(repoRoot, 'code', moduleName)
  if (!fs.existsSync(modulePath)) {
    findings.push(`${moduleName}: audited module does not exist`)
    continue
  }
  for (const source of filesUnder(path.join(modulePath, 'tests'))) {
    if (!trackedSources.has(source)) findings.push(`${source}: missing test contract entry`)
  }
}

if (findings.length > 0) {
  console.error('Test contract check failed:')
  for (const finding of findings) console.error(`- ${finding}`)
  process.exit(1)
}

const active = manifest.tests.filter((test) => test.status === 'active').length
const legacy = manifest.tests.length - active
console.log(
  `Test contract check passed: ${manifest.tests.length} tests, ` +
  `${active} active owners, ${legacy} tracked migration debts`,
)
