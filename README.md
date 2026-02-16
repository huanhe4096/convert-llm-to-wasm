# WebGPU Text Embedding Demo (Vue + Bun + Vite)

An online text embedding demo built with `transformers.js`, designed for large-scale input (up to 100k sentences) with UMAP 2D visualization.

## Features

- Model: default `mixedbread-ai/mxbai-embed-xsmall-v1` (editable in UI)
- Inference: `transformers.js` + `WebGPU`
- Data generation: random sentence generation for testing
- Progress UI: model loading / embedding / UMAP fit / UMAP transform
- Timing: total elapsed time in milliseconds
- UMAP strategy (Embedding Atlas style):
  - Fit on a random sample of `8192` points
  - Transform remaining points in batches of `16384`
  - Incremental point-cloud updates during transform
- Dimensionality truncation: `384 / 256 / 128` (default `128`) to reduce downstream UMAP cost
- Non-blocking UX: embedding + UMAP run inside a Web Worker

## Tech Stack

- `bun`
- `vite`
- `vue 3`
- `@huggingface/transformers`
- `umap-js`

## Run

```bash
bun install
bun dev
```

Build:

```bash
bun run build
```

## UI Layout

- Two-column layout
- Left column:
  - Controls (model / sentence count / embedding batch size / used dim)
  - Text input
  - Progress and timing
- Right column:
  - UMAP 2D point cloud (incremental updates)

## Core Workflow

1. Main thread collects settings + sentences and sends them to the Worker.
2. Worker runs embedding in batches.
3. Once `8192` sampled embeddings are ready, Worker runs UMAP `fit` immediately (no need to wait for all embeddings).
4. Remaining embeddings are transformed in `16384` batches.
5. Each transform batch is posted back and rendered incrementally.

## Important Files

- `/Users/hh667/workspace/test/webgpu-text-embedding/src/App.vue`: UI + Worker messaging
- `/Users/hh667/workspace/test/webgpu-text-embedding/src/workers/embedding.worker.js`: embedding + UMAP pipeline
- `/Users/hh667/workspace/test/webgpu-text-embedding/src/style.css`: two-column layout and visualization styles

## Notes

- Truncating to 256/128 reduces UMAP/visualization work, but does not reduce model forward hidden-size computation.
- For very large datasets, browser memory/VRAM can still become the bottleneck. Use smaller batch sizes and lower target dimensions as needed.

## Prompt Template (for similar projects)

Use this structure when asking an LLM to generate a similar demo.

### 1) Goal

- One-sentence target outcome.
- Example: `Build an online text embedding + UMAP visualization app with transformers.js, scalable to 100k sentences.`

### 2) Hard Constraints

- Required stack (e.g., `JavaScript + bun + vite + vue`)
- Required model ID
- Runtime requirement (e.g., `WebGPU required`)

### 3) Functional Requirements

- Data input / random sample generation
- Embedding batching
- Progress + elapsed time
- UMAP strategy:
  - `fit(sample=8192)`
  - `transform(batch=16384)`
  - incremental visualization updates
- Dimensionality options (`384/256/128`)

### 4) Non-functional Requirements

- No main-thread blocking (use Worker)
- Scale target (100k)
- Maintainable and configurable code

### 5) UI Requirements

- Two-column layout:
  - Left: controls and progress
  - Right: point cloud

### 6) Deliverables

- Runnable code
- List of changed files
- Run instructions
- Implementation summary and limitations

### 7) Verification Requirements

- Run: `bun run build`
- Report: build status and any warnings

## Example Prompt

```text
Build an online text embedding demo with the following requirements:
1. Stack: JavaScript + bun + vite + vue.
2. Model: mixedbread-ai/mxbai-embed-xsmall-v1 using transformers.js + WebGPU.
3. Random sentence generator with max 100000 sentences.
4. Embedding batch support (default 100, configurable).
5. Show progress stages (loading, embedding, umap-fit, umap-transform) and total elapsed time.
6. UMAP strategy: fit on random 8192, then transform remaining points in 16384 batches; append each transform batch to visualization.
7. Support dimension truncation options (384/256/128, default 128).
8. Run embedding + UMAP inside a Web Worker; keep main thread for UI/render only.
9. Use a two-column UI: left panel for controls/progress, right panel for point cloud.
10. Return changed file list and run bun run build for verification.
```

## Deploy To GitHub Pages (Bun + Vite)

This project is configured for GitHub Pages deployment under a repo subpath:

- Expected URL format: `https://<user>.github.io/<repo>/`
- The build `base` is set from `BASE_PATH` (the workflow sets it automatically to `/<repo>/`).

### One-time GitHub setup

1. In your GitHub repo, go to `Settings -> Pages`.
2. Set `Source` to `GitHub Actions`.

### Deploy

Push to `main` and GitHub Actions will build and deploy automatically using:

- Workflow: `/Users/hh667/workspace/convert-llm-to-wasm/.github/workflows/deploy.yml`
- Base path during CI build: `/${GITHUB_REPOSITORY#*/}/`
