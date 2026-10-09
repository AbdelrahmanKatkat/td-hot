---
name: model-type-workflow
description: Research and define a model-specific preprocessing, model-output decoding, geospatial postprocessing, training-parameter, and inference-parameter contract for a new fAIr model or helper library. Use before implementing a new model type or when its tensor I/O or postprocessing behavior is not yet established.
---

# Model-Type Workflow

## Goal

Produce a concise, evidence-backed model-type specification that implementation and tests can follow. This skill is architecture-agnostic and does not prescribe repository layout; use `repo-structure-workflow` for that.

## Working rules

- Research the specific model/task rather than assuming behavior from the model family name.
- Prefer primary sources: the model's paper, official implementation, model card, export documentation, and official dataset/task documentation.
- Distinguish **verified facts**, **proposed design decisions**, and **unknowns**. Cite links for externally researched claims.
- Inspect available checkpoints, export code, or existing implementation when possible. Never infer tensor shapes, output semantics, normalization, or coordinate conventions from a model name alone.
- Keep model-space decoding independent of geospatial metadata and map-space georeferencing independent of architecture-specific output tensor layout.
- If required information is unavailable, list the unknown and propose a small verification experiment rather than presenting a guess as fact.

## Inputs to establish

Collect or explicitly mark as unknown:

1. **Task:** one of `semantic-segmentation`, `instance-segmentation`, `object-detection`, or `classification`.
2. **Expected geometry:** `polygon`, `line`, or `point`; explain how the task's predictions map to that geometry. Do not force an unnatural geometry without a documented product requirement.
3. **Input contract:** three-band RGB OAM; channel order; accepted source dtype/value range; NCHW tensor shape; tensor dtype; fixed/dynamic spatial dimensions.
4. **Preprocessing:** resize, letterbox, crop/tiling policy, padding, normalization, interpolation, and handling of chip dimensions not divisible by model stride. Research what the model needs as it's case sensitive.
5. **Output contract:** every output tensor's name/order, shape, dtype, semantic meaning, coordinate frame, stride/downsampling factor, and whether outputs are logits, probabilities, boxes, masks, or embeddings. Research what the model needs as it's case sensitive.
6. **Labels and geospatial assumptions:** label representation, chip/label alignment, CRS, affine transform, pixel convention, and any known source imagery constraints.
7. **Weights and export:** checkpoint source and license, export command/tooling, supported ONNX opset/runtime, and known export limitations.

## Research procedure

1. Search official model documentation, repository code, paper/model card, and ONNX/export documentation.
2. Confirm preprocessing and output semantics from code/configuration where possible, not just paper descriptions.
3. Record exact tensor shapes and dimensions, noting whether they are fixed, dynamic, or dependent on image size.
4. Trace coordinates through resize/crop/padding and output stride so predictions can be mapped back to original chip pixels.
5. Identify which postprocessing operations are model-space operations and which require map-space/geospatial context.
6. Define parameters with defaults, types, safe ranges, and an explanation of what each parameter changes.
7. Identify questions that require a small local test or an integration check.
8. Produce the deliverable below before implementing substantial model-specific logic.

## Deliverable: model-type specification

Use concise bullets and tables, not long prose.

### 1. Task and output contract

- Task type and expected geometry.
- Intended GeoJSON FeatureCollection feature schema.
- Class IDs/names and confidence semantics.
- Assumptions, limitations, and unknowns.

### 2. Preprocessing pipeline

Document the exact sequence from RGB chip to model tensor:

- Input band/channel order, shape, dtype, and value range.
- Resize/letterbox/crop/tiling policy and interpolation.
- Normalization formula and channel statistics, if applicable.
- Output tensor shape/layout/dtype and contiguous-memory expectations if relevant.
- Padding/crop offsets and scale factors.
- Metadata required for inverse mapping: original width/height, processed dimensions, resize scale, padding/crop offsets, affine transform, CRS, and any other information needed to recover source-pixel positions.

State which details are verified and which need testing.

### 3. Model-output interpretation (model space)

For every output tensor, specify:

- Name/order, shape, dtype, and semantics.
- Activation: sigmoid, softmax, argmax, or none.
- Thresholding, class selection, NMS, mask assembly, connected components, skeletonization, or other decoding operations as relevant.
- Box/mask/point coordinate convention and output stride.
- Decoded representation passed to geospatial postprocessing.

This section must not depend on CRS, affine transforms, or ground units.

### 4. Geospatial postprocessing (map space)

Specify:

- Mapping from model coordinates back to original chip-pixel coordinates.
- Polygonization, box-to-polygon conversion, line tracing, or point extraction as appropriate.
- Geometry validity and cleanup: minimum area/length, hole handling, simplification tolerance, merge/dissolve rules, and whether thresholds are pixel-based or ground-unit-based.
- CRS handling and conversion to EPSG:4326 for GeoJSON.
- Feature properties, such as `class_id`, `class_name`, `confidence`, and `source`, with definitions and types.
- Empty predictions, invalid geometries, and edge-touching objects behavior.

Do not apply area/length thresholds in physical units until CRS and units are known. Avoid silently changing geometry semantics.

### 5. Training parameters

For each user-facing parameter, specify:

| Key | Type | Default | Safe range/options | Purpose |
|---|---|---|---|---|

Include relevant `training.*` keys and required fAIr split parameters `val_ratio` and `split_seed`. Explain device selection: use CUDA only when available/visible and otherwise CPU, unless the established training contract specifies a different explicit behavior. Serving should use CPU ONNX inference unless the platform contract says otherwise.

Distinguish required parameters from optional tuning knobs. Do not expose a parameter merely because an underlying library has one.

### 6. Inference parameters

Use the same table format. Include `inference.*` keys and identify the stage each parameter affects:

- preprocessing,
- model-space decoding,
- map-space geometry cleanup.

Include confidence thresholds, class filters, NMS/IoU thresholds, and geometry cleanup parameters only when relevant to the task. Ensure the empty-parameter case runs end-to-end with safe defaults.

### 7. Export and verification plan

- ONNX input/output names, shapes, dtypes, opset, and dynamic axes.
- CPU runtime requirements and unsupported operators, if known.
- Numerical parity checks between source model and ONNX.
- Shape and basic numeric tests.
- Geospatial round-trip tests using synthetic/local fixtures.
- Offline test plan; no network downloads in unit tests.
- Required changes to the consuming fAIr model pack's STAC shapes and pipeline behavior.

### 8. Evidence and open questions

List key sources with URLs and the specific claims they support. Separate:
- Verified from primary sources or repository code.
- Proposed implementation choices.
- Unresolved questions and the test needed to resolve each one.

## Parameter design rules

- Every user-facing parameter needs a default, type, validation rule, and clear stage/meaning.
- `parse_*` functions must tolerate `None` and missing keys.
- Support known fAIr aliases such as `confidence_threshold` when necessary; define precedence if both canonical and alias keys are present.
- Keep Python parser keys and STAC `mlm:hyperparameters` / `fair:hyperparameters_spec` aligned. Follow the actual consuming repository's conventions for prefixes and bare keys.
- When adding a parameter, test both default behavior and explicit override behavior.
- Do not invent numerical defaults or safe ranges without evidence or a clearly labelled proposal.

## Stop conditions

Do not start architecture-specific implementation as if the contract were settled when any of these remain unknown:
- Input normalization or channel order.
- ONNX output tensor semantics or spatial scale.
- Coordinate mapping through resize/padding/crops.
- Required output geometry or feature-property contract.

Instead, document the unknown and identify the smallest experiment or source inspection needed to resolve it.
