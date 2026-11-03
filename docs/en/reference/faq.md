# FAQ

This page answers the most common questions that repeat across the main line.

## 1. Build-related

### 1.1 Why does `chapter00` run locally but later chapters fail to build?

Usually because later chapters introduce more specific targets, headers, or library dependencies. First identify whether the failure is from the compiler, linker, or path setup, then go back to the corresponding chapter `README` and `CMakeLists.txt`.

### 1.2 Why does `docs:build` still show an old error after I changed the file?

The usual causes are cache, anchor mismatch, or a site-layer path issue. First run `npm run docs:check-links`, then check whether the markdown file is actually mapped into the site entry, and finally verify that the page links point to existing headings and files.

## 2. Runtime-related

### 2.1 Why does `chapter00` work but `chapter01` fails?

`chapter00` mainly checks the environment and the first baseline path. `chapter01` starts loading weights and building the object model. If `chapter01` fails, check the model file, loader, and `Tensor` shape, dtype, and ownership.

### 2.2 Why does the same chapter work on CPU but fail on GPU?

This usually means the CPU path and CUDA path have different prerequisites. Check whether you are still on the CPU-first path or have already switched to the GPU backend, then decide whether the failure is environment-related, build-related, or implementation-related.

## 3. Memory-related

### 3.1 Why does memory usage keep growing during generation?

The most common reason is that cache, buffer, or intermediate results are not being reused as expected. Check the `KV Cache`, scheduler, and request lifecycle in `chapter06_generation` first.

### 3.2 Why does quantization still consume a lot of memory?

Quantization usually lowers weight storage cost first, but it does not automatically solve activation, cache, or temporary buffer usage. Check the quantization scope in `chapter05_quantization`, then see whether the runtime still needs optimization.

## 4. Verification-related

### 4.1 Why can I not tell whether the issue is logic or benchmark noise?

Separate correctness tests from benchmark measurements. Fix logic first, then measure performance. If you mix them, the result is usually unstable.

### 4.2 Why should I read the chapter `intro.md` first?

The `intro.md` shows the chapter's position, learning content, source order, and verification criteria. Chapter conclusions and verification steps are kept there instead of in separate `summary.md` pages.

## 5. How to use this page

- Find the question category first
- Then return to the relevant chapter `intro.md`
- If it is still unclear, read the detailed chapter page or the reference layer

## 6. Maintenance rule

- Keep questions short and answers pointed
- Do not expand long theory here
- When a new FAQ appears, prefer adding it to the nearest main-line chapter first
