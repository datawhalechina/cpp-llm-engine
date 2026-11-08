#!/usr/bin/env python3
"""Sync shared markdown sections between paired documentation files.

The repository has a few docs that share the same logical sections but differ in
their surrounding site-specific framing. This script keeps those shared sections
in sync by replacing matching `##` sections in the target with the source
content.

Add more pairs to PAIRS as needed. A pair can also map different heading names
via `section_map`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class SectionSpec:
    source_heading: str
    target_heading: str | None = None

    @property
    def effective_target_heading(self) -> str:
        return self.target_heading or self.source_heading


@dataclass(frozen=True)
class DocPair:
    source: Path
    target: Path
    sections: tuple[SectionSpec, ...]
    rewrites: tuple[tuple[str, str], ...] = ()


REPO_ROOT = Path(__file__).resolve().parents[1]

PAIRS: tuple[DocPair, ...] = (
    DocPair(
        source=REPO_ROOT / "README.md",
        target=REPO_ROOT / "docs" / "README.md",
        sections=(
            SectionSpec("这套教程的定位"),
            SectionSpec("和常见推理引擎的对照"),
            SectionSpec("章节总览"),
            SectionSpec("读法建议"),
            SectionSpec("本地预览"),
            SectionSpec("相关入口"),
        ),
        rewrites=(
            (r"\]\(source/([^)]+?)\.md\)", r"](/\1)"),
            (r"\]\(docs/README\.md\)", r"](/README)"),
            (r"\]\(docs/README\.en\.md\)", r"](/en/README)"),
            (r"\]\(docs/guide\.md\)", r"](/guide)"),
            (r"\]\(USAGE\.md\)", r"](/guide)"),
            (r"\]\(source/guides/README\.md\)", r"](/guides/README)"),
            (r"\]\(source/reference/README\.md\)", r"](/reference/README)"),
            (r"\]\(source/maintenance/README\.md\)", r"](/maintenance/README)"),
        ),
    ),
)


def split_sections(text: str) -> tuple[str, list[tuple[str, list[str]]]]:
    lines = text.splitlines(keepends=True)
    prefix: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_heading is None and not sections:
                prefix = current_lines
            elif current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = line.rstrip("\n")
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_heading is None:
        prefix = current_lines
    else:
        sections.append((current_heading, current_lines))

    return ("".join(prefix), sections)


def build_section_map(sections: Iterable[tuple[str, list[str]]]) -> dict[str, list[str]]:
    return {heading: lines for heading, lines in sections}


def replace_sections(source_text: str, target_text: str, specs: tuple[SectionSpec, ...]) -> str:
    source_prefix, source_sections = split_sections(source_text)
    target_prefix, target_sections = split_sections(target_text)

    source_map = build_section_map(source_sections)
    replacement_map = {
        f"## {spec.effective_target_heading}": source_map.get(f"## {spec.source_heading}")
        for spec in specs
    }

    updated_sections: list[tuple[str, list[str]]] = []
    for heading, lines in target_sections:
        replacement = replacement_map.get(heading)
        if replacement is not None:
            if replacement is None:
                raise KeyError(heading)
            updated_sections.append((heading, replacement))
        else:
            updated_sections.append((heading, lines))

    result: list[str] = [target_prefix]
    for _, lines in updated_sections:
        result.extend(lines)

    return "".join(result)


def apply_rewrites(text: str, rewrites: tuple[tuple[str, str], ...]) -> str:
    for pattern, replacement in rewrites:
        text = re.sub(pattern, replacement, text)
    return text


def sync_pair(pair: DocPair) -> bool:
    source_text = pair.source.read_text(encoding="utf-8")
    target_text = pair.target.read_text(encoding="utf-8")
    updated_text = replace_sections(source_text, target_text, pair.sections)
    updated_text = apply_rewrites(updated_text, pair.rewrites)

    if updated_text == target_text:
        print(f"ok: {pair.target} already in sync")
        return False

    pair.target.write_text(updated_text, encoding="utf-8")
    print(f"updated: {pair.target}")
    return True


def main() -> int:
    changed = False
    for pair in PAIRS:
        changed |= sync_pair(pair)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
