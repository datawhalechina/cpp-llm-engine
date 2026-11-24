#!/usr/bin/env node
// 防止多模态构建重新直接编译 Loader、量化或语言模型源码。
import fs from 'node:fs'
import path from 'node:path'

const repoRoot = process.cwd()
const multimodalCmakePath = path.join(repoRoot, 'code/multimodal/CMakeLists.txt')
const modelCmakePath = path.join(repoRoot, 'code/model/CMakeLists.txt')
const multimodalCmake = fs.readFileSync(multimodalCmakePath, 'utf8')
const modelCmake = fs.readFileSync(modelCmakePath, 'utf8')
const findings = []

const forbiddenSourceRoots = [
  '../chapter01_tensor_loader/src',
  '../chapter05_quantization/src',
  '../model/src',
]
for (const sourceRoot of forbiddenSourceRoots) {
  if (multimodalCmake.includes(sourceRoot)) {
    findings.push(`code/multimodal/CMakeLists.txt directly references ${sourceRoot}`)
  }
}

const requiredMultimodalTargets = [
  'chapter01_tensor_loader',
  'chapter05_quantization',
  'cpp_llm_engine_llama_cpu_model',
  'cpp_llm_engine_llama_cuda_model',
  'chapter01_real_tokenizer',
]
for (const target of requiredMultimodalTargets) {
  if (!multimodalCmake.includes(target)) {
    findings.push(`code/multimodal/CMakeLists.txt misses reusable target ${target}`)
  }
}

const focusedModelSources = [
  'src/real_llama_cpu_decoder.cpp',
  'src/cuda_llama_layer_weights.cpp',
  'src/real_llama_cuda_model.cpp',
]
for (const source of focusedModelSources) {
  const occurrences = modelCmake.split(source).length - 1
  if (occurrences !== 1) {
    findings.push(`code/model/CMakeLists.txt registers ${source} ${occurrences} times`)
  }
}

for (const option of [
  'CHAPTER01_BUILD_STANDALONE_TARGETS',
  'CHAPTER05_BUILD_STANDALONE_TARGETS',
  'CPP_LLM_MODEL_BUILD_STANDALONE_TARGETS',
]) {
  if (!multimodalCmake.includes(`set(${option} OFF)`)) {
    findings.push(`multimodal dependency setup does not disable ${option}`)
  }
}

if (findings.length > 0) {
  console.error('Multimodal target boundary check failed:')
  for (const finding of findings) console.error(`- ${finding}`)
  process.exit(1)
}

console.log('Multimodal target boundary check passed.')
