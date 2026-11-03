#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/migrate_chapter_to_source.sh <chapter-dir>

Examples:
  scripts/migrate_chapter_to_source.sh chapter03_matmul
  scripts/migrate_chapter_to_source.sh chapter06_generation

What it does:
  - Moves docs/<chapter-dir> to source/<chapter-dir> if needed
  - Replaces docs/<chapter-dir> with a symlink pointing to ../source/<chapter-dir>
  - Is safe to rerun when the chapter has already been migrated
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

chapter_dir="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docs_path="$repo_root/docs/$chapter_dir"
source_path="$repo_root/source/$chapter_dir"
docs_parent_rel="../source/$chapter_dir"

if [[ ! "$chapter_dir" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: invalid chapter dir name: $chapter_dir" >&2
  exit 1
fi

mkdir -p "$repo_root/source"

if [[ -L "$docs_path" ]]; then
  current_target="$(readlink "$docs_path")"
  if [[ "$current_target" == "$docs_parent_rel" ]]; then
    echo "ok: docs/$chapter_dir already points to source/$chapter_dir"
    exit 0
  fi
  echo "error: docs/$chapter_dir is already a symlink to $current_target" >&2
  exit 1
fi

if [[ -d "$docs_path" ]]; then
  if [[ -e "$source_path" ]]; then
    echo "error: both docs/$chapter_dir and source/$chapter_dir exist; resolve manually" >&2
    exit 1
  fi
  mv "$docs_path" "$source_path"
  ln -s "$docs_parent_rel" "$docs_path"
  echo "migrated: docs/$chapter_dir -> source/$chapter_dir"
  exit 0
fi

if [[ -d "$source_path" ]]; then
  ln -s "$docs_parent_rel" "$docs_path"
  echo "linked: docs/$chapter_dir -> source/$chapter_dir"
  exit 0
fi

echo "error: neither docs/$chapter_dir nor source/$chapter_dir exists" >&2
exit 1
