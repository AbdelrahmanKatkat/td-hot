# `td-hot`

Python helper library for the HOT **tree-crown detector** on fAIr. It wraps the
[`weecology/deepforest-tree`](https://huggingface.co/weecology/deepforest-tree) model (DeepForest, RetinaNet
ResNet-50-FPN) so a fAIr model pack (`fAIr-models/models/<pack>/`) can stay a thin STAC/ZenML/Docker adapter.

**Python:** >= 3.12. Model architecture, preprocessing, postprocessing, sliding window and spatial split are
documented in [`theory.md`](theory.md).

## Inference path

```text
chip dir (GeoTIFF RGB)
  -> sliding_windows()           400 px windows, 5 % overlap (DeepForest semantics)
  -> preprocess_window()         ImageNet normalise, resize short side to 800, pad to 32
  -> onnxruntime session.run()   cls_logits, bbox_regression (provided by fAIr)
  -> decode_retinanet_output()   anchors, box decode, score filter, per-window NMS
  -> window_to_chip() + merge_window_detections()   chip pixels, cross-window NMS
  -> detections_to_feature_collection()             EPSG:4326 box polygons
```

Entry point: `td_hot.predict_session(session, input_images, params)` returns a GeoJSON `FeatureCollection`
(`confidence`, `class_id`, `class_name`, `source` per feature). All `params` are optional; see `theory.md` 6.1.

## Install and export the ONNX model

```bash
uv sync                          # inference core: numpy, Pillow, pyproj, rasterio
uv sync --extra train --extra test
uv run python -m td_hot.export weights/NEON.pt deepforest_tree.onnx   # needs the `train` extra
```

ONNX I/O: input `images (1,3,H,W)` float32 with dynamic `H`, `W`; outputs `cls_logits (1,A,1)`,
`bbox_regression (1,A,4)`. Resize, normalisation, anchors, decoding and NMS live in numpy, not in the graph.

## Layout

| Module | Role |
|---|---|
| `params.py` | defaults + `parse_*` (tolerate `None`, aliases `score_thresh`, `nms_thresh`) |
| `preprocess.py` | windows, resize/normalise/pad, `PreprocessMeta` |
| `decode.py` | anchors, RetinaNet decode, NMS, cross-window merge (no CRS) |
| `georef.py`, `postprocess.py` | window -> chip -> EPSG:4326, GeoJSON |
| `serve.py` | `predict_session`, `predict_chip` |
| `dataset.py` | `spatial_split` (block split on OAM tile coordinates) |
| `export.py` | ONNX export (`train` extra, import explicitly) |

## Tests

```bash
uv run ruff check src tests
uv run pytest -q                      # offline unit tests
uv run pytest -m integration          # opt-in: torchvision/ONNX parity, needs weights/NEON.pt + train extra
```

## Not done yet

- Fine-tuning (dataset conversion, training, evaluation) is not implemented; `val_ratio`, `split_seed` and
  `block_size` are parsed and `spatial_split` exists for it.
- The fAIr-models pack (`pipeline.py`, `stac-item.json`, Docker) is not written, so integration with the
  fAIr-models validator is unverified. Pin this library there by Git tag or commit SHA.
- Accuracy on OpenAerialMap imagery (not 10 cm NEON) is unmeasured; see `theory.md` 8.3.
