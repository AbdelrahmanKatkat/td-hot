# `td-hot`

Python helper library for the HOT **tree-crown detector** on fAIr. It wraps the
[`weecology/deepforest-tree`](https://huggingface.co/weecology/deepforest-tree) model (DeepForest, RetinaNet
ResNet-50-FPN) so a fAIr model pack (`fAIr-models/models/<pack>/`) can stay a thin STAC/ZenML/Docker adapter.

**Python:** >= 3.12. Model architecture, preprocessing, postprocessing and spatial split are documented in
[`docs/Theory.md`](docs/Theory.md); the datasets behind it are analysed in `research/datasets.md` (local, not committed).

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
(`confidence`, `class_id`, `class_name`, `source` per feature). All `params` are optional; see `docs/Theory.md` 6.1.

**Chip names must be `OAM-<x>-<y>-<z>.<ext>`.** The zoom and tile row give the true ground resolution, so the
scale is right even for PNG/JPEG chips with no georeferencing (their polygons stay in pixel coordinates). A
zoom that would need an upscale above 16x is rejected. The scaled chip is also capped at `max_input_px` (default 1024) on its longest
side, so zoom 18 is upscaled 4x instead of 11x and its memory stays bounded; zoom 20 (about 10-15 cm/px, about 3x) is the level to
recommend and is not affected.

## Install, extras and who installs what

| Extra | Contents | Who installs it |
|---|---|---|
| *(none)* | numpy, Pillow, pyproj, rasterio. **No torch.** | the CPU inference image (plus `onnxruntime`, which the fAIr serving base provides) |
| `train` | torch, torchvision, onnx: fine-tuning, evaluation, ONNX export | the fAIr **GPU** training image |
| `infer` | onnxruntime | running an exported ONNX locally (`scripts/predict.py`); the fAIr serving base brings its own |
| `test` | onnxruntime, pytest | CI |
| `dataprep` | datasets (Hugging Face) | **this repository only** (`research/analysis`); fAIr-models must never install it |

Training always runs on a GPU, but the CUDA build of torch is chosen by the image (for example
`--extra-index-url https://download.pytorch.org/whl/cu126`), not by this package's metadata. Inference is CPU only.
`onnxscript` is not required: the export uses the legacy exporter (`dynamo=False`).

```bash
uv sync                                # inference core
uv sync --extra train --extra test     # development
uv run python -m td_hot.export weights/NEON.pt NEON.onnx    # base-model ONNX (needs `train`)
```

A fAIr model pack installs it from a pinned ref, for example
`uv pip install "td-hot[train] @ git+https://github.com/<owner>/td-hot@<tag-or-sha>"` in the training stage and
`"td-hot @ git+https://github.com/<owner>/td-hot@<tag-or-sha>"` in the inference stage. The repository has to be
reachable without credentials, and the ref must exist.

> **Local CUDA environment:** if you put a CUDA build of torch into this repo's `.venv`, `uv sync` and a plain
> `uv run` put the CPU build back. Use `uv run --no-sync ...` or `.venv/Scripts/python ...`.

ONNX I/O ("level 2"): input `image (3,H,W)` float32 in [0, 1] with dynamic `H`, `W`; outputs `boxes (N,4)`,
`scores (N)`, `labels (N)`, best score first, before NMS. Normalisation, padding, the network, anchors, the top
1000 per level, box decoding and clipping are inside the graph; the confidence filter and NMS (both tunable) are
in `decode_detections`. The third-party ONNX in the first STAC item (fixed 256 px input, NMS inside) has a
different contract and is not accepted: export the base model with the command above.

## Layout

| Module | Role |
|---|---|
| `params.py` | defaults + `parse_*` (tolerate `None`, aliases `score_thresh`, `nms_thresh`, `lr`) |
| `preprocess.py` | the one scale step, `PreprocessMeta` |
| `tiles.py` | OAM chip name -> ground resolution -> scale (zoom and latitude, no CRS) |
| `decode.py` | confidence filter, NMS, cap on the ONNX candidates (no CRS) |
| `georef.py`, `postprocess.py` | resized -> chip -> EPSG:4326, GeoJSON |
| `serve.py` | `predict_session`, `predict_chip` |
| `dataset.py` | `spatial_split` (block split on OAM tile coordinates), GeoJSON boxes -> per-chip pixel boxes |
| `export.py` | `build_retinanet` (owns normalisation, anchors, thresholds), level-2 ONNX export (`train`) |
| `train.py` | RetinaNet fine-tuning from `NEON.pt` (`train`) |
| `evaluate.py` | precision / recall / F1 at one or several IoUs, confidence sweep (`train`) |

`train`, `evaluate` and `export` are imported explicitly (`from td_hot.train import ...`); `import td_hot` never
loads torch.

## Scripts

`scripts/` holds thin command-line entry points; each one only parses arguments and calls `td_hot`.

| Script | Calls | Extra |
|---|---|---|
| `scripts/finetune.py` | `td_hot.train` + `export`: fine-tunes, writes `<name>.pt` and `<name>.onnx`, `train.log`, `history.json` | `train` |
| `scripts/evaluate.py` | `td_hot.evaluate`: P/R/F1 at several IoUs, confidence sweep, and a reference-image score for the forgetting check | `train` |
| `scripts/predict.py` | `td_hot.predict_session`: an ONNX over a folder of chips to GeoJSON, on CPU | `infer` |

```bash
python scripts/finetune.py --train-dir data/guru_boxes/train --val-dir data/guru_boxes/val --weights weights/NEON.pt     --out data/runs/full --name NEON_OAM --epochs 3 --batch-size 4
python scripts/evaluate.py --weights data/runs/full/NEON_OAM.pt --chips data/guru_boxes/val/chips     --labels data/guru_boxes/val/labels.json --sweep
python scripts/predict.py --onnx data/runs/full/NEON_OAM.onnx --chips data/guru_boxes/test/chips --out predictions.geojson
```

Dataset-specific code (the Guru conversion and analysis) stays in `research/`, not in `src/` or `scripts/`.

## Fine-tuning

```python
import logging
from td_hot import labels_for_chips, parse_split_params, spatial_split
from td_hot.params import parse_preprocess_params, parse_train_params
from td_hot.train import train_retinanet
from td_hot.evaluate import score_chips, sweep_confidence
from td_hot.export import export_onnx_bytes

logging.basicConfig(level=logging.INFO)   # optional: train_retinanet makes its own progress visible anyway
labels = labels_for_chips("chips", "labels")              # OAM-x-y-z.tif chips + one .geojson -> pixel boxes, nothing copied
split = parse_split_params({})
train_names, val_names = spatial_split(sorted(labels), split.val_ratio, split.split_seed, block_size=split.block_size)
res = train_retinanet(
    train_images_dir="chips", train_labels=labels, train_names=train_names,
    val_images_dir="chips", val_labels=labels, val_names=val_names,
    cfg=parse_train_params({"epochs": 5}), pre=parse_preprocess_params({}),   # same scale as serving
    checkpoint_path="weights/NEON.pt",
    on_epoch_end=lambda epoch, train_loss, val_loss: ...)    # e.g. forward to MLflow
print(score_chips(res.model, images_dir="chips", labels_json=labels, names=val_names))   # validation chips only
rows = sweep_confidence(res.model, images_dir="chips", labels_json=labels, names=val_names)
onnx_bytes = export_onnx_bytes(res.model)
```

**Recipe** (defaults of `TrainParams`, the recipe of the fAIr `deepforest_tree` pack): SGD momentum 0.9, learning rate
1e-3, weight decay 1e-4, cosine schedule (per batch, optional `warmup_steps`), gradient clipping at 1.0, horizontal-flip augmentation, **the ResNet encoder
frozen** (`freeze_encoder`) and BatchNorm always frozen. Freezing the encoder is what limits catastrophic forgetting of
the NEON features; set `freeze_encoder=False` only with plenty of target data and check the source-domain score.
The checkpoint stores weights only: normalisation, anchors and thresholds come from `build_retinanet`, so every
generation of a model must be rebuilt through it.

**Logging.** Progress goes to the `td_hot.train` logger: a start summary (device, sizes, recipe), loss lines with an
ETA, one line per epoch (including peak GPU memory), a warning if it runs on a CPU. The number of step lines is
bounded at about 50 per epoch whatever the dataset size (`log_every` overrides it). If the host process configured no
logging (a bare container step), `train_retinanet` attaches a stdout handler to its own logger so the lines are not
silent; a host that configured logging (ZenML, `basicConfig`) is left alone. `predict_session` logs one summary line.

Crown polygons become their bounding boxes. Metrics are greedy-matched P/R/F1 at an IoU threshold (not mAP).
A freshly fine-tuned model is usually less confident than the base model: choose `confidence_threshold` with
`sweep_confidence` rather than keeping 0.1.

## From a pipeline step (what a fAIr-models pack calls)

A step trains straight from the original chips and GeoJSON: no prepared folder is copied or shared between steps.

```python
from td_hot import parse_split_params, parse_preprocess_params, parse_postprocess_params, spatial_split
from td_hot.dataset import labels_for_chips
from td_hot.params import parse_train_params
from td_hot.train import train_retinanet                  # `train` extra, imported inside the step
from td_hot.export import export_onnx_bytes
from td_hot.evaluate import score_session                 # no torch: scores the exported ONNX

split = parse_split_params(hyperparameters)               # val_ratio, split_seed, block_size
labels = labels_for_chips(chips_dir, labels_geojson_dir)  # chip name -> pixel boxes
train_names, val_names = spatial_split(sorted(labels), split.val_ratio, split.split_seed, block_size=split.block_size)
result = train_retinanet(
    train_images_dir=chips_dir, train_labels=labels, train_names=train_names,
    val_images_dir=chips_dir, val_labels=labels, val_names=val_names,
    cfg=parse_train_params(hyperparameters), pre=parse_preprocess_params(hyperparameters),
    checkpoint_path=base_weights, on_epoch_end=lambda epoch, train_loss, val_loss: ...)  # -> MLflow
onnx_bytes = export_onnx_bytes(result.model)
metrics = score_session(onnx_session, images_dir=chips_dir, labels=labels, names=val_names)[0.5]
```

At serving time the pack calls `td_hot.predict_session(session, input_images, params)`; `td_hot.serve.detect_chip`
returns the detections in chip pixels when a step needs them instead of GeoJSON.

## Tests

```bash
uv run ruff check src tests
uv run pytest -q                      # offline unit tests
uv run pytest -m integration          # opt-in: torchvision/ONNX parity and a train/eval smoke test; needs weights/NEON.pt + train extra
```

## Not done yet

- A full fine-tune on the Guru dataset (only short pilots have been run) and accuracy on other OpenAerialMap
  regions; see `docs/Theory.md` 8.3.
- The existing fAIr-models pack `models/deepforest_tree` was written against an earlier version of this library and
  has to be updated (see `research/fair-models-gap.md`); the ref it pins must exist in this repository.
