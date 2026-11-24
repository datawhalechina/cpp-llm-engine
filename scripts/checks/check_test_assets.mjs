#!/usr/bin/env node
// 检查 C++/CUDA 测试是否完整接入 CMake、CTest 和公共测试配置。
import fs from 'node:fs'
import path from 'node:path'

const codeRoot = 'code'
const findings = []

function filesUnder(root, predicate) {
  const files = []
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const fullPath = path.join(root, entry.name)
    if (entry.isDirectory()) files.push(...filesUnder(fullPath, predicate))
    if (entry.isFile() && predicate(fullPath)) files.push(fullPath)
  }
  return files
}

function cmakeCommands(text, commandName) {
  const commands = []
  const pattern = new RegExp(`\\b${commandName}\\s*\\(`, 'g')
  for (const match of text.matchAll(pattern)) {
    let depth = 1
    let cursor = match.index + match[0].length
    let quoted = false
    while (cursor < text.length && depth > 0) {
      const character = text[cursor]
      if (character === '"' && text[cursor - 1] !== '\\') quoted = !quoted
      if (!quoted && character === '(') depth += 1
      if (!quoted && character === ')') depth -= 1
      cursor += 1
    }
    if (depth === 0) commands.push(text.slice(match.index, cursor))
  }
  return commands
}

function firstArgument(command) {
  return command.slice(command.indexOf('(') + 1, -1).trim().split(/\s+/)[0]
}

const cmakeFiles = filesUnder(codeRoot, (file) => path.basename(file) === 'CMakeLists.txt')
const testSources = filesUnder(codeRoot, (file) =>
  file.includes(`${path.sep}tests${path.sep}`) && /(?:test|tests)\.(?:cpp|cu)$/.test(file),
)

for (const source of testSources) {
  const moduleRoot = source.slice(0, source.indexOf(`${path.sep}tests${path.sep}`))
  const cmakeFile = path.join(moduleRoot, 'CMakeLists.txt')
  if (!fs.existsSync(cmakeFile)) {
    findings.push(`${source}: module has no CMakeLists.txt`)
    continue
  }
  const relativeSource = path.relative(moduleRoot, source).split(path.sep).join('/')
  const cmakeText = fs.readFileSync(cmakeFile, 'utf8')
  if (!cmakeText.includes(relativeSource)) {
    findings.push(`${source}: not referenced by ${cmakeFile}`)
  }
}

let executableTestCount = 0
let ctestCount = 0
for (const cmakeFile of cmakeFiles) {
  const text = fs.readFileSync(cmakeFile, 'utf8')
  const executableCommands = cmakeCommands(text, 'add_executable')
  const addTestCommands = cmakeCommands(text, 'add_test')
  const prepareCommands = cmakeCommands(text, 'cpp_llm_prepare_test')
  const labelCommands = cmakeCommands(text, 'cpp_llm_set_test_labels')
  const propertyCommands = cmakeCommands(text, 'set_tests_properties')

  const ctestNames = new Set()
  for (const command of addTestCommands) {
    const match = command.match(/\bNAME\s+([^\s)]+)/)
    if (match && !match[1].includes('${')) ctestNames.add(match[1])
  }
  ctestCount += ctestNames.size

  const preparedTargets = new Set(prepareCommands.map(firstArgument))
  const labelledTests = new Set(labelCommands.map(firstArgument))
  for (const command of propertyCommands) {
    if (!/\bLABELS\b/.test(command)) continue
    const body = command.slice(command.indexOf('(') + 1, -1).trim()
    const testNames = body.split(/\bPROPERTIES\b/)[0].trim().split(/\s+/)
    for (const testName of testNames) labelledTests.add(testName)
  }

  for (const command of executableCommands) {
    if (!/\btests\//.test(command)) continue
    const target = firstArgument(command)
    if (target.includes('${')) continue
    executableTestCount += 1
    if (!preparedTargets.has(target)) {
      findings.push(`${cmakeFile}: test target ${target} misses cpp_llm_prepare_test`)
    }
    if (!ctestNames.has(target)) {
      findings.push(`${cmakeFile}: test target ${target} is not registered with add_test`)
    }
    if (!labelledTests.has(target)) {
      findings.push(`${cmakeFile}: CTest ${target} has no labels`)
    }
  }

  for (const testName of ctestNames) {
    if (!labelledTests.has(testName)) {
      findings.push(`${cmakeFile}: CTest ${testName} has no labels`)
    }
  }

  const hasGenericTestFactory = /add_executable\(\$\{target\}/.test(text)
  if (hasGenericTestFactory) {
    for (const required of [
      /cpp_llm_prepare_test\(\$\{target\}\)/,
      /add_test\(NAME\s+\$\{target\}/,
      /cpp_llm_set_test_labels\(\$\{target\}/,
    ]) {
      if (!required.test(text)) {
        findings.push(`${cmakeFile}: generic test factory misses a required integration step`)
      }
    }
  }
}

if (findings.length > 0) {
  console.error('Test asset check failed:')
  for (const finding of findings) console.error(`- ${finding}`)
  process.exit(1)
}

console.log(
  `Test asset check passed: ${testSources.length} sources, ` +
    `${executableTestCount} executable targets, ${ctestCount} literal CTest entries`,
)
