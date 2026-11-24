#!/usr/bin/env node

import { spawnSync } from 'node:child_process'
import {
  existsSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
} from 'node:fs'
import { dirname, extname, join, resolve } from 'node:path'

const distDir = 'docs/.vitepress/dist'
rmSync(distDir, { recursive: true, force: true })

const result = spawnSync('npm', ['run', 'docs:build'], {
  stdio: 'inherit',
})

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}

if ((result.status ?? 1) !== 0) {
  process.exit(result.status ?? 1)
}

const htmlFiles = []

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) {
      walk(path)
    } else if (path.endsWith('.html')) {
      htmlFiles.push(path)
    }
  }
}

function walkMarkdown(dir, output) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) {
      walkMarkdown(path, output)
    } else if (path.endsWith('.md')) {
      output.push(path)
    }
  }
}

const sourceMarkdownFiles = []
walkMarkdown('source', sourceMarkdownFiles)

const codeIndexMarkdownFiles = []
walkMarkdown('docs/code', codeIndexMarkdownFiles)

const repositoryMarkdownFiles = [...sourceMarkdownFiles, ...codeIndexMarkdownFiles]
walkMarkdown('code', repositoryMarkdownFiles)
walkMarkdown('exercises', repositoryMarkdownFiles)
walkMarkdown('docs/en', repositoryMarkdownFiles)
for (const file of [
  'README.md',
  'USAGE.md',
  'CONTRIBUTING.md',
  'SESSION_HANDOFF.md',
  'docs/index.md',
]) {
  if (existsSync(file)) repositoryMarkdownFiles.push(file)
}

const linkMissing = []
const markdownImagePattern = /!\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)/g
const markdownLinkPattern = /(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)/g
const legacyCodeLinkPattern =
  /href="(?:\.\.\/)+_code\/|href="\/_code\//
const repositoryLinkPattern =
  /https:\/\/github\.com\/datawhalechina\/cpp-llm-engine\/(?:blob|tree)\/main\/([^\s)"'<>#?]+)(?:[?#][^\s)"'<>]*)?/g

function localLinkTarget(file, link) {
  const [rawPath] = link.split(/[?#]/)
  if (!rawPath) return null

  const normalized = decodeURIComponent(
    rawPath.startsWith('<') && rawPath.endsWith('>')
      ? rawPath.slice(1, -1)
      : rawPath,
  )
  const target = resolve(dirname(realpathSync(file)), normalized)
  const candidates = [target]
  if (!extname(target)) {
    candidates.push(`${target}.md`, join(target, 'README.md'), join(target, 'index.md'))
  }
  return candidates.find((candidate) => existsSync(candidate)) ?? target
}

let checkedLocalLinks = 0
let checkedRepositoryLinks = 0
for (const file of repositoryMarkdownFiles) {
  const markdown = readFileSync(file, 'utf8')
  if (legacyCodeLinkPattern.test(markdown)) {
    linkMissing.push(
      `${file}: legacy _code link cannot be opened from GitHub source view`,
    )
  }
  for (const match of markdown.matchAll(markdownImagePattern)) {
    const image = match[1]
    if (/^(https?:|data:)/.test(image)) continue

    const target = localLinkTarget(file, image)
    if (!existsSync(target)) {
      linkMissing.push(`${file}: image ${image} -> ${target}`)
    }
  }
  for (const match of markdown.matchAll(markdownLinkPattern)) {
    const link = match[1]
    if (/^(https?:|mailto:|tel:|data:|javascript:|#|\/)/.test(link)) continue

    checkedLocalLinks += 1
    const target = localLinkTarget(file, link)
    if (!existsSync(target)) {
      linkMissing.push(`${file}: Markdown ${link} -> ${target}`)
    }
  }
  for (const match of markdown.matchAll(repositoryLinkPattern)) {
    checkedRepositoryLinks += 1
    const linkedPath = decodeURIComponent(match[1])
    const target = resolve(linkedPath)
    if (!existsSync(target)) {
      linkMissing.push(`${file}: repository ${linkedPath} -> ${target}`)
    }
  }
}

if (linkMissing.length > 0) {
  console.error('\nBroken repository documentation links:')
  for (const item of linkMissing) {
    console.error(`- ${item}`)
  }
  process.exit(1)
}

function hrefToFile(href, fromFile) {
  if (!href || href.startsWith('#')) return null
  if (/^(https?:|mailto:|tel:|data:)/.test(href)) return null

  const [rawPath] = href.split('#')
  if (!rawPath || rawPath.startsWith('javascript:')) return null

  const normalized = decodeURIComponent(rawPath)

  const basePath = normalized.startsWith('/')
    ? join(distDir, normalized.slice(1))
    : resolve(dirname(fromFile), normalized)

  if (normalized === '/') return join(distDir, 'index.html')
  if (normalized.endsWith('/')) return join(basePath, 'index.html')
  if (extname(basePath)) return basePath
  return `${basePath}.html`
}

walk(distDir)

const missing = []
const hrefPattern = /\bhref="([^"]+)"/g
const imagePattern = /\bsrc="([^"]+)"/g

for (const file of htmlFiles) {
  const html = readFileSync(file, 'utf8')
  for (const match of html.matchAll(hrefPattern)) {
    const target = hrefToFile(match[1], file)
    if (target && !existsSync(target)) {
      missing.push(`${file}: ${match[1]} -> ${target}`)
    }
  }
  for (const match of html.matchAll(imagePattern)) {
    const target = hrefToFile(match[1], file)
    if (target && !existsSync(target)) {
      missing.push(`${file}: image ${match[1]} -> ${target}`)
    }
  }
}

if (missing.length > 0) {
  console.error('\nBroken generated internal links:')
  for (const item of missing) {
    console.error(`- ${item}`)
  }
  process.exit(1)
}

console.log(
  `Checked ${repositoryMarkdownFiles.length} Markdown files, ${checkedLocalLinks} local links, ${checkedRepositoryLinks} repository links, and ${htmlFiles.length} generated HTML files.`,
)
