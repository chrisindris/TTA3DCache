# Submodule Mapping

This repository keeps the upstream forks under `src/tta3dcache/cdViews`, `src/tta3dcache/TDA`, and `src/tta3dcache/Uni-Adapter` as references. The production TTA3DCache implementation lives in the top-level `src/tta3dcache` package and wraps the submodules instead of copying them wholesale.

## cdViews references

- [src/tta3dcache/cdViews/cdviews/qa_inference.py](../src/tta3dcache/cdViews/cdviews/qa_inference.py) is the host inference loop and the main reference for view ranking, NMS, prompting, answer parsing, and JSON output.
- [src/tta3dcache/cdViews/cdviews/qa_utils.py](../src/tta3dcache/cdViews/cdviews/qa_utils.py) is the reference for dataset loading and config merging.
- [src/tta3dcache/cdViews/cdviews/dataset.py](../src/tta3dcache/cdViews/cdviews/dataset.py) is the reference for scene/question examples and view metadata.
- [src/tta3dcache/cdViews/cdviews/ViewSelector.py](../src/tta3dcache/cdViews/cdviews/ViewSelector.py) and [src/tta3dcache/cdViews/cdviews/train.py](../src/tta3dcache/cdViews/cdviews/train.py) are the references for the learned view selector.
- [src/tta3dcache/cdViews/cdviews/view_distance_calculation.py](../src/tta3dcache/cdViews/cdviews/view_distance_calculation.py) is the reference for spatial view-distance computation.

Ported behavior:

- stable wrapper interfaces for prepared examples and candidate selections;
- baseline prompt and answer flow abstractions;
- view-set generation and logging around the existing ranked view lists.

Not ported:

- the full upstream model-loading stack;
- training code for the selector;
- direct dependence on the upstream CLI layout.

## TDA references

- [src/tta3dcache/TDA/tda_runner.py](../src/tta3dcache/TDA/tda_runner.py) is the reference for fixed-size cache updates and cache-logit fusion.
- [src/tta3dcache/TDA/utils.py](../src/tta3dcache/TDA/utils.py) is the reference for entropy, confidence, and evaluation helpers.

Ported behavior:

- fixed-size answer cache with confidence-aware pruning;
- cache scoring with optional similarity weighting;
- entropy-aware fallback logic.

Not ported:

- class-label classification assumptions;
- positive/negative cache training loops;
- dataset-specific CLIP evaluation code.

## Uni-Adapter references

- [src/tta3dcache/Uni-Adapter/Uni_Adapter.py](../src/tta3dcache/Uni-Adapter/Uni_Adapter.py) is the main reference for confidence weighting, EMA-style prototype updates, and cache fusion.
- [src/tta3dcache/Uni-Adapter/utils/hyperparams.py](../src/tta3dcache/Uni-Adapter/utils/hyperparams.py) is the reference for dataset-dependent cache hyperparameters.
- [src/tta3dcache/Uni-Adapter/utils/math_utils.py](../src/tta3dcache/Uni-Adapter/utils/math_utils.py) is the reference for online refinement and smoothing utilities.
- [src/tta3dcache/Uni-Adapter/main_test-time.py](../src/tta3dcache/Uni-Adapter/main_test-time.py) is the reference for the test-time pipeline structure and logging setup.

Ported behavior:

- EMA-style answer prototypes;
- confidence and uncertainty proxies;
- conservative fusion between base predictions and cached evidence.

Not ported:

- point-cloud task assumptions;
- graph smoothing as an MVP requirement;
- training-time components.

## Implementation boundary

The new code under `src/tta3dcache` reimplements the required abstractions cleanly and keeps the upstream forks as reference material only. The CPU-only smoke tests exercise the new wrapper and cache logic without requiring external checkpoints.
