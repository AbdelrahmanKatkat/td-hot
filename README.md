# `td-hot`

Python helper library for the HOT **tree-crown detector** on fAIr. It wraps the
[`weecology/deepforest-tree`](https://huggingface.co/weecology/deepforest-tree) model (DeepForest, RetinaNet
ResNet-50-FPN) so a fAIr model pack (`fAIr-models/models/<pack>/`) can stay a thin STAC/ZenML/Docker adapter.

**Python:** >= 3.12. Model architecture, preprocessing, postprocessing, sliding window and spatial split are
documented in [`theory.md`](theory.md).

## Inference path

```text
chip dir (OAM-<x>-<y>-<z>.tif/.png/.jpg, RGB)
  -> preprocess_chip()           one resize; scale = chip ground resolution (from the zoom and tile row) / 0.05 m
  -> onnxruntime session.run()   ONNX level 2: normalise, network, anchors, decode -> boxes, scores, labels
  -> decode_detections()         confidence filter, NMS, cap 300
  -> detections_to_chip()        back to chip pixels
  -> detections_to_feature_collection()             EPSG:4326 box polygons
```

Entry point: `td_hot.predict_session(session, input_images, params)` returns a GeoJSON `FeatureCollection`
(`confidence`, `class_id`, `class_name`, `source` per feature). All `params` are optional; see `theory.md` 6.1.

**Chip names must be `OAM-<x>-<y>-<z>.<ext>`.** The zoom and tile row give the true ground resolution, so the
scale is right even for PNG/JPEG chips with no georeferencing (their polygons stay in pixel coordinates). A
zoom that would need an upscale above 8x is rejected. Zoom 20 (about 10-15 cm/px) is the level to recommend.

## Install and export the ONNX model

```bash
uv sync                          # inference core: numpy, Pillow, pyproj, rasterio
uv sync --extra train --extra test
uv run python -m td_hot.export weights/NEON.pt deepforest_tree.onnx   # needs the `train` extra
```

ONNX I/O ("level 2"): input `image (3,H,W)` float32 in [0, 1] with dynamic `H`, `W`; outputs `boxes (N,4)`,
`scores (N)`, `labels (N)`, best score first, before NMS. Normalisation, padding, the network, anchors, the top
1000 per level, box decoding and clipping are inside the graph; the confidence filter and NMS (both tunable) are
in `decode_detections`. The third-party ONNX in the shared STAC item has a different contract (fixed 256 px
input, NMS inside) and is not accepted: export the base model with the command above.

## Layout

| Module | Role |
|---|---|
| `params.py` | defaults + `parse_*` (tolerate `None`, aliases `score_thresh`, `nms_thresh`) |
| `preprocess.py` | the one scale step, `PreprocessMeta` |
| `tiles.py` | OAM chip name -> ground resolution -> scale (zoom and latitude, no CRS) |
| `decode.py` | confidence filter, NMS, cap on the ONNX candidates (no CRS) |
| `georef.py`, `postprocess.py` | resized -> chip -> EPSG:4326, GeoJSON |
| `serve.py` | `predict_session`, `predict_chip` |
| `dataset.py` | `spatial_split` (block split on OAM tile coordinates), GeoJSON crowns -> per-chip pixel boxes |
| `export.py` | `build_retinanet` (owns normalisation, anchors, thresholds), level-2 ONNX export (`train` extra, import explicitly) |
| `train.py` | RetinaNet fine-tuning from `NEON.pt` (`train` extra, import explicitly) |
| `evaluate.py` | precision / recall / F1 at IoU 0.5 through the same level-2 + `decode_detections` path (`train` extra) |

## Fine-tuning

```python
from td_hot.dataset import prepare_detection_dataset_from_geojson
from td_hot.params import parse_preprocess_params, parse_split_params, parse_train_params
from td_hot.train import train_retinanet
from td_hot.evaluate import evaluate_retinanet
from td_hot.export import export_onnx_bytes

prepare_detection_dataset_from_geojson(   # OAM-x-y-z.tif chips + one crown .geojson; out_dir must be empty
    chips_dir="chips", labels_dir="labels", out_dir="prepared", split_cfg=parse_split_params({}))
res = train_retinanet(
    train_images_dir="prepared/train/images", train_labels_json="prepared/train/labels.json",
    val_images_dir="prepared/val/images", val_labels_json="prepared/val/labels.json",
    cfg=parse_train_params({"epochs": 5}), pre=parse_preprocess_params({}),   # same scale as serving
    checkpoint_path="weights/NEON.pt")
print(evaluate_retinanet(res.model, images_dir="prepared/val/images", labels_json="prepared/val/labels.json"))
onnx_bytes = export_onnx_bytes(res.model)
```

Crown polygons become their bounding boxes. BatchNorm statistics stay frozen during training. Metrics are
greedy-matched P/R/F1 at IoU 0.5, not mAP. Only a random-weight smoke test (`pytest -m integration`) and a
one-step check on `NEON.pt` have been run; no real fine-tune on OAM data has been done.

## Tests

```bash
uv run ruff check src tests
uv run pytest -q                      # offline unit tests
uv run pytest -m integration          # opt-in: torchvision/ONNX parity, needs weights/NEON.pt + train extra
```

## Not done yet

- A real fine-tune on labelled OAM chips, and the resulting metrics, have not been run.
- The fAIr-models pack (`pipeline.py`, `stac-item.json`, Docker) is not written, so integration with the
  fAIr-models validator is unverified. Pin this library there by Git tag or commit SHA.
- Accuracy on OpenAerialMap imagery (not 10 cm NEON) is unmeasured; see `theory.md` 8.3.
