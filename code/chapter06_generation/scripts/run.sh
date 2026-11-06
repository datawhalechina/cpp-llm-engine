#!/usr/bin/env bash
# Chapter 06 兼容入口：转发到正式的单请求 Generation Core 验证。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_generation_core.sh" "$@"
