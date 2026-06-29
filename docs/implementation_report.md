# Implementation Report

## Files changed

- Added the root packaging and typed configuration.
- Added answer normalization, equivalence, confidence, candidate-view generation, cache, prototype, scoring, integration, pipeline, evaluation, and logging modules.
- Added CLI scripts for baseline, TTA3DCache, evaluation, and inspection.
- Added configuration presets and the submodule mapping note.

## Design decisions

- Kept the MVP training-free and CPU-friendly.
- Implemented the core pipeline as pure Python modules with optional torch-backed tensors.
- Preserved cdViews, TDA, and Uni-Adapter as references instead of merging them into one code path.
- Used deterministic generation and deterministic tie-breaking throughout.

## Deviations from the specification

- The real cdViews runtime is wrapped by stable interfaces, but the smoke path uses a mock adapter and mock VLM so the repository can be tested without external weights.
- The full embedding-based answer clustering and feature encoder are pluggable; the default CPU test path uses deterministic fallbacks.

## Unresolved issues

- The upstream cdViews script still depends on its original runtime environment and is not executed in the unit tests.
- The root baseline CLI currently demonstrates the integration contract with mocked inputs rather than the external benchmark dataset.

## Commands used for tests

- `python -m unittest discover -s tests`

## Test results

- All 29 unit tests passed.

## Next recommended experiment

- Wire the `StaticCdViewsAdapter` to a real ranked-view export from cdViews and run the baseline and full pipeline on a small validation subset.
