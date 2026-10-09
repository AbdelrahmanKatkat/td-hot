---
name: repo-structure-workflow
description: Design, bootstrap, or validate the repository structure and integration contract for a reusable fAIr helper library or a fAIr base-model pack. Use when creating a repo, changing packaging or public API, splitting inference and training dependencies, or coordinating changes with fAIr-models.
---

# Repository-Structure Workflow

## Goal

Create or maintain a repository that is reproducible, testable, compatible with fAIr, and easy to extend without unnecessary abstraction or accidental public API breaks.

This skill covers repository boundaries, layout, packaging, verification, and integration. Use `model-type-workflow` for architecture/task research.

## First identify the repository type

Determine which target is being changed:

- **Helper library:** reusable model logic imported by one or more model packs.
- **fAIr base-model pack:** adapter inside `fAIr-models` that wires STAC, ZenML, Docker, training/inference steps, and the platform serving contract.

Do not mix the two layouts or move model-pack responsibilities into a helper library.

## A. Helper-library repository

### Responsibilities

The helper library owns reusable logic such as:

- RGB chip preprocessing and metadata needed to reverse spatial transforms.
- ONNX output decoding into model-space masks/detections/class maps.
- Georeferencing, vectorization, and GeoJSON FeatureCollection creation.
- Parameter dataclasses and robust dictionary parsing.
- Optional dataset preparation, training, evaluation, and ONNX export.

The consuming fAIr model pack remains a thin adapter for STAC metadata, ZenML wiring, Docker, and the platform contract.

### Recommended layout

Adapt to the existing repository and avoid creating empty modules just to match a template.

```text
<repo>/
├── CLAUDE.md
├── pyproject.toml
├── README.md
├── src/
│   └── <package_name>/
│       ├── __init__.py
│       ├── params.py
│       ├── preprocess.py
│       ├── decode.py
│       ├── postprocess.py
│       ├── georef.py
│       ├── serve.py
│       ├── dataset.py              
│       ├── train.py                
│       ├── evaluate.py             
│       └── export.py               
├── tests/
│   ├── test_params.py
│   ├── test_preprocess.py
│   ├── test_decode.py
│   ├── test_postprocess.py
│   └── test_serve.py
└── .claude/
    └── skills/
        ├── model-type-workflow/
        │   └── SKILL.md
        └── repo-structure-workflow/
            └── SKILL.md
```

Module boundaries are recommendations, not mandatory filenames. Prefer fewer well-defined modules over splitting every helper into its own file.

### Packaging rules

- Set `requires-python = ">=3.12"`.
- Keep core dependencies limited to packages required by the inference path.
- Put training/export-only dependencies in a `train` extra, such as `.[train]`.
- Put pytest and test-only dependencies in a `test` extra.
- Classify each new dependency explicitly: inference core, training/export extra, or test extra.
- Do not import heavyweight training frameworks at module import time in the serving path.
- Confirm that packaging includes the intended `src/<package_name>` modules and any bundled non-code assets.
- Keep dependency versions and build backend consistent with the current project; do not change packaging tools without a reason.

### Public API and compatibility

- Expose the supported API through `src/<package_name>/__init__.py`.
- Public API should normally include task-appropriate `preprocess_*`, `decode_*`, and `postprocess_*` functions, `predict_session(session, input_images, params)`, and parameter parsers/dataclasses.
- `preprocess_*` returns `(batch_tensor, meta)` with sufficient spatial and georeferencing metadata.
- `decode_*` converts ONNX outputs to model-space representation without CRS/affine dependencies.
- `postprocess_*` maps decoded results to geospatial features and returns a GeoJSON FeatureCollection.
- Do not promise that every model type uses identical tensor signatures; document the supported stable contract and task-specific differences.
- Before renaming/removing/moving a public symbol, search for consumers and tests. Update the consuming model pack in the same coordinated change or retain a compatibility shim.
- Test empty/missing parameter dictionaries and explicit overrides.

### Offline testing

Unit tests must not download weights, datasets, or imagery. Use synthetic chips, tiny generated masks, local fixtures, and mocks.

Cover, as applicable:
- parameter defaults, aliases, invalid values, and overrides;
- RGB validation and deterministic tensor shape/dtype/range;
- decode behavior for representative synthetic ONNX-like outputs;
- inverse scaling and padding/crop offsets;
- georeferencing and CRS conversion;
- valid FeatureCollection output, feature properties, and empty predictions;
- serving-path imports without training extras installed;
- source-model/ONNX parity in a separately controlled integration test if local test assets are available.

A unit test should not silently skip online or download resources to succeed.

## B. fAIr base-model pack in `fAIr-models`

Create or update `models/<model_name>/` with the files required by the current repository contract. The established pack structure commonly includes:

- `pipeline.py`
- `stac-item.json`
- `README.md`
- `Dockerfile`
- `Dockerfile.dockerignore`
- `tests/conftest.py`
- `tests/test_steps.py`

Verify this list against the current validator and `docs/contributing/model.md`; repository requirements may evolve.

Contract details to verify against the consumer repository:
- Required ZenML pipeline/step exports and exact signatures.
- Required training and ONNX artifact names and serialization types.
- STAC task, input/output, hyperparameter, metrics, and asset fields.
- Absolute HTTPS URLs for assets that require remote URLs.
- Docker build targets, inference image dependencies, and step-test expectations.

Prefer a thin adapter: keep reusable architecture/model logic in the helper library and orchestration/platform glue in `pipeline.py`.

## C. Integration and pinning

When helper-library behavior or public API changes:

1. Identify every consuming model pack available in the workspace.
2. Decide whether the change is backward-compatible.
3. If breaking, add a compatibility shim where feasible or coordinate the pack update in the same change.
4. Pin the consumer to a Git tag or commit SHA; do not rely on a moving branch for reproducible builds.
5. Update the consuming pack's imports, dependency pin, preprocessing/output contract, and STAC shapes where relevant.
6. In `fAIr-models`, run the model validator and the relevant step tests. Use the repository's current documented commands.
7. Record the new library ref and any migration notes in the relevant README/changelog.

Prefer version tags such as `vX.Y.Z`. For breaking changes, bump the major version and document migration.

If the consuming repository is not available, do not claim integration was verified. Report the exact checks that remain.

## D. Verification loop

From the helper-library repository root, run:

```bash
ruff check src tests
ruff format src tests
uv run pytest -q
```

Inspect each exit code and output. `ruff format src tests` modifies files; review the resulting diff.

For relevant changes, also verify:
- the package can be installed with core dependencies only;
- the `train` extra installs the training/export stack;
- tests work offline;
- public imports work from the installed package;
- the consuming model pack works with the pinned ref.

From the `fAIr-models` repository, use the current repository-specific validation flow. Where applicable, commands may include:

```bash
just validate
python -m scripts.validate_model models/<model_name>
ruff check models/<model_name>
ruff format --check models/<model_name>
```

Do not assume these commands or targets exist in a different repository. Inspect its docs and available scripts first.

## E. Final review checklist

Before declaring the task complete:

- [ ] Repository responsibilities remain separated.
- [ ] Core dependencies are inference-safe; optional dependencies are in the correct extras.
- [ ] Public exports and consumer imports agree.
- [ ] RGB input, tensor shape/dtype, output semantics, and geospatial metadata are tested.
- [ ] Model-space decoding is independent of geospatial metadata.
- [ ] Map-space processing is independent of ONNX output layout details.
- [ ] Tests require no network access.
- [ ] STAC and consuming-pack changes are coordinated where relevant.
- [ ] Validation results and remaining integration work are reported accurately.
