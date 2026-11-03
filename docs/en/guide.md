# Reading Guide

This page tells you how to move through the tutorial.

The Chinese main line is now in a closed state. This page keeps the route map
and reading rules up to date while the current work shifts to English chapter
bodies and residual cleanup.

## Main Line

This tutorial moves along one inference-engine main line instead of being a loose collection of topics.

| Stage | Main responsibility |
| --- | --- |
| `chapter01` - `chapter03` | inputs, tensor semantics, and operator boundaries |
| `chapter04` - `chapter06` | structure and runtime |
| `chapter05` | representation and cost |
| `chapter07` | verification and evidence |

After `chapter07`, continue to profiling and serving, then use `reference` and `guides` as needed:

- `reference`: concept explanations, terminology, and ecosystem comparison
- `guides`: build, environment, migration, and engineering methods

If you want to understand the "control tower / engine" split first, read [training vs inference frameworks](/en/reference/framework_comparison). That page shows where Python frameworks, C++ engines, and the engine layer sit in the system.

If you only want the high-level map, this page is enough.
If you want chapter-by-chapter details, go back to [the Chinese main entry](/README) or [the English entry](/en/README).
If you want the shortest route map, start from [the English entry](/en/README).

## Three quick rules

- foundation prepares you for the main line.
- chapter00_quickstart gets you to the first working baseline.
- chapter01 to chapter07 form the main teaching path.

## Recommended path

1. Start from [README.md](README) to see the English entry.
2. If your background is thin, read [Foundation](foundation/intro) first.
3. If you want a runnable baseline first, start with [chapter00_quickstart](chapter00_quickstart/intro).
4. If you need engineering help, open [guides/README.md](guides/README).
5. If you want design context, open [reference/README.md](reference/README).

## CPU / GPU split

- Chapter 00 to Chapter 02 are CPU-first and easy to verify.
- Chapter 03 is where GPU becomes a clear backend.
- Chapter 03's cuBLAS section is the first concrete GPU implementation point.

## Where to look next

- [Foundation](foundation/intro)
- [chapter00_quickstart](chapter00_quickstart/intro)
- [guides/README.md](guides/README)
- [reference/README.md](reference/README)

## Daily checks

If you are changing docs or links, run:

```bash
npm run docs:check-links
```

Then run:

```bash
npm run docs:build
```
