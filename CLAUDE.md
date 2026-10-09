# fAIr Helper Library — Claude Code Instructions

## Purpose and architecture

This repository is a reusable Python helper library for model logic used by an fAIr base-model pack in `fAIr_repos/fAIr-models/models/<pack>/`.

Keep responsibilities separate:
- **This library:** reusable preprocessing, model-output decoding, georeferencing/vectorization, parameter parsing, and training/evaluation/export logic.
- **The fAIr model pack:** thin adapter for STAC metadata, ZenML pipeline wiring, Docker images, and the fAIr serving contract.

The platform-facing inference path must accept RGB imagery and produce a GeoJSON `FeatureCollection`.

## Before changing code

1. Inspect the repository layout, `pyproject.toml`, public exports, relevant implementation, and tests before editing. If found.
2. For model-type or architecture-specific decisions, use `.claude/skills/model-type-workflow/SKILL.md`.
3. For repository layout, packaging, or fAIr integration work, use `.claude/skills/repo-structure-workflow/SKILL.md`.
4. Search for existing implementations and reuse established patterns. Do not assume a package, API, model output, or fAIr contract without checking it.
5. If this library is consumed by a model pack, inspect the consuming pack when available before changing public APIs or I/O behavior.

## Non-negotiable compatibility rules

- **RGB only:** the supported imagery input is three-band RGB. Validate or clearly document channel order, shape, dtype, and value range.
- **Vector output:** `predict_session(session, input_images, params)` must return a valid GeoJSON `FeatureCollection` for the supported inference task.
- **Lightweight serving:** core dependencies must support inference only. Put training/export dependencies behind a `train` extra and test-only dependencies behind a `test` extra where practical.
- **Stable public API:** expose the supported API through `src/<package_name>/__init__.py`. If a public symbol must move or be renamed, update the consuming model pack in the same change or retain a compatibility shim.
- **Offline unit tests:** tests must not download weights, datasets, imagery, or other network resources. Use tiny generated fixtures, local test assets, or mocks.
- **Model-space/map-space separation:** decoding model outputs must not depend on CRS, affine transforms, or ground units. Georeferencing and vectorization must not depend on architecture-specific ONNX tensor layouts.
- **Deterministic preprocessing:** define and test the input tensor layout, shape, dtype, normalization, resize/padding behavior, and metadata required to map results back to source-chip coordinates.
- **Safe parameter defaults:** parsers must tolerate `None` and missing keys, apply documented defaults, and support established fAIr aliases when needed (for example, `confidence_threshold`).
- **No speculative abstractions:** make the smallest maintainable change. Avoid unnecessary dependencies, configuration knobs, layers, or refactors.

## Public API contract

Keep a small, documented public surface, normally exported from `src/<package_name>/__init__.py`:

- `preprocess_*`: returns `(batch_tensor, meta)`; `meta` carries the source dimensions and georeferencing/rescaling information required downstream.
- `decode_*`: converts raw ONNX outputs into a model-space representation such as a class mask or detections.
- `postprocess_*`: converts the decoded representation and metadata into a GeoJSON `FeatureCollection`.
- `predict_session(session, input_images, params)`: main inference entry point.
- `params.py`: dataclasses and `parse_*` functions for preprocessing, inference, postprocessing, training, and data splitting where applicable.
- Optional training/export API: dataset preparation, training, evaluation, and ONNX export helpers. Keep these optional and separate from the inference import path.

Confirm exact signatures and types against the existing implementation and consuming model pack before changing them.

## Packaging and dependencies

- `requires-python` must be `>=3.12`.
- Keep core dependencies minimal and inference-safe.
- Put training/export dependencies (for example, PyTorch, Transformers, Ultralytics, and ONNX tooling as actually required) in the `train` extra.
- Put pytest and test-only dependencies in the `test` extra.
- Before adding a dependency, state whether it is required at inference time, training/export time, or test time. Add it to the corresponding dependency group.
- Ensure importing the package's inference API does not require importing heavy training frameworks.

## Change workflow

1. Identify the behavior and public contracts affected.
2. Inspect callers, especially the fAIr model pack, before changing public API or tensor I/O.
3. Implement the smallest coherent change.
4. Add focused offline tests for defaults, overrides, shapes, numeric behavior, georeferencing, and output schema as relevant.
5. Run the verification commands below; if a command is unavailable or fails, report that honestly.
6. Review the diff for accidental API breaks, generated files, unrelated edits, and dependency changes.
7. If a consuming model pack exists, update its pinned library ref and validate the integration in the `fAIr-models` repository.

## Verification

From the helper-library repository root:

```bash
ruff check src tests
ruff format src tests
uv run pytest -q
```

These commands are not substitutes for checking results: inspect exit codes and test output. Do not claim that a command passed unless it was actually run and its result was checked.

For changes affecting ONNX input/output, preprocessing, public API, or fAIr integration, add the relevant integration checks described in `.claude/skills/repo-structure-workflow/SKILL.md`.

## fAIr-models integration and releases

- Model packs should install this library from a pinned Git tag or commit SHA, not a moving branch.
- When a released library change affects a consumer, update the consuming model pack's pin in the same coordinated change where possible.
- Prefer version tags such as `vX.Y.Z` for reproducibility.
- For breaking public API changes, bump the major version, provide migration guidance, and update consumers or retain compatibility shims.
- In the consuming `fAIr-models` repository, run the relevant model validator and model-pack step tests.

## Completion report

At the end of a task, report:
- What changed and why.
- Tests and validation commands actually run, with results.
- Any tests that could not run and why.
- Public API, dependency, model-pack pin, or release implications.
- Remaining limitations or unverified assumptions.
