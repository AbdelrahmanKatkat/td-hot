# Theory: DeepForest tree-crown detector (`weecology/deepforest-tree`)

Contract for the `td-hot` library. Tags: **[V]** verified from source code, the checkpoint, or a run in this repo;
**[P]** proposed design decision; **[?]** unknown / needs an experiment (section 8).

## 0. Summary

| Item | Value |
|---|---|
| Task / geometry | Object detection, one class `Tree` (id 0). Output is an axis-aligned **box polygon** per tree crown [V] |
| Model | torchvision **RetinaNet, ResNet-50 + FPN**, 1 class, 32,254,189 parameters [V] |
| Input | 3-band RGB, designed for **~10 cm** imagery, run on **400x400 px windows** [V: paper, DeepForest config] |
| Inference | Sliding window (400 px, 5 % overlap) -> RetinaNet -> per-window NMS -> cross-window NMS [V] |
| Weights | HF `weecology/deepforest-tree` (MIT per the model card); local copy `weights/NEON.pt` is the same network as a plain state dict [V] |
| This repo | ONNX holds backbone+FPN+head only. Resize, normalization, anchors, box decoding, NMS, windowing and georeferencing are numpy/Python [P] |

## 1. Task and output contract

- Task `object-detection`, geometry `polygon` (each box becomes a closed 5-point ring, EPSG:4326). Crowns are boxes, not outlines: the model never predicts a crown shape [V].
- GeoJSON `FeatureCollection`; per-feature `properties`:

| Property | Type | Meaning |
|---|---|---|
| `confidence` | float, 4 dp | sigmoid score of the RetinaNet classification logit, in (0.1, 1] with defaults |
| `class_id` | int | always `0` |
| `class_name` | str | `"Tree"` (HF `config.json` `label_dict = {"Tree": 0}`) |
| `source` | str | chip file name |

- Empty prediction -> `{"type": "FeatureCollection", "features": []}` [V, tested].
- Chips without a CRS/transform (plain PNG/JPEG) produce polygons in **pixel coordinates**, not lon/lat. fAIr chips are GeoTIFFs, so this only affects local experiments [V].

## 2. Model architecture

### 2.1 Network (verified by loading `NEON.pt` with `strict=True` into `torchvision.models.detection.retinanet_resnet50_fpn(num_classes=1)`)

```
RGB (N,3,H,W), ImageNet-normalised, H and W multiples of 32
 └─ ResNet-50 body (conv1 7x7/2 -> maxpool -> layer1..layer4)       C3, C4, C5 are used (strides 8, 16, 32)
     └─ FPN (256 ch): inner_blocks x3 (1x1), layer_blocks x3 (3x3)  P3, P4, P5
         └─ extra_blocks p6, p7 (3x3, stride 2)                     P6, P7 (strides ~64, ~128)
             ├─ classification head: 4x(conv3x3+ReLU) -> cls_logits  (9, 256, 3, 3)  = 9 anchors x 1 class
             └─ regression head:     4x(conv3x3+ReLU) -> bbox_reg    (36, 256, 3, 3) = 9 anchors x 4 deltas
```

- 301 state-dict keys, prefixes `backbone.body`, `backbone.fpn`, `head.classification_head`, `head.regression_head` [V].
- **Anchors** (torchvision default, used by DeepForest): per level sizes `(32,40,50) (64,80,101) (128,161,203) (256,322,406) (512,645,812)` x aspect ratios `(0.5, 1, 2)` = 9 per location. Sizes are **truncated to int** (`int(x * 2**(1/3))`); using the exact cube roots shifts anchors by up to 1 px, which was caught by the parity test [V].
- **Loss** (training only): focal loss for classification and L1 for box regression, as in RetinaNet [Lin et al. 2017] [V: paper cites it as the detector; loss is torchvision's].
- ResNet-101/152 were tried in the paper and gave no gain, so ResNet-50 is final [V: paper 2.3].

### 2.2 Transform inside torchvision that DeepForest does **not** override [V]

`GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=ImageNet, image_std=ImageNet)`:

1. normalise `(x/255 - mean) / std` with mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`;
2. bilinear resize with `scale = min(800 / min(h, w), 1333 / max(h, w))`, new size `floor(h*scale)`;
3. zero-pad bottom/right to a multiple of 32.

Consequence: a 400x400 window is **upscaled 2x to 800x800** before the network, and a 256x256 fAIr chip is upscaled 3.125x. This is verified in the DeepForest source (no `min_size` is passed to `RetinaNet`) and by the parity tests (section 7).

### 2.3 Training approach behind the checkpoint (from the paper and model card)

- Stage 1: a **LiDAR-based unsupervised** tree detector produces millions of moderate-quality boxes on NEON RGB tiles; the RGB RetinaNet is pre-trained on them [V: Weinstein et al. 2019, sec. 2.2-2.3; HF card].
- Stage 2: fine-tune on a small hand-annotated set. Reported at the San Joaquin NEON site: recall 0.69 / precision 0.60 at IoU 0.5 for the full model, against < 20 % recall for the LiDAR algorithm alone [V: paper abstract/conclusions].
- The model card also cites a cross-site follow-up (Ecological Informatics 2020) and the DeepForest package paper (Methods in Ecology and Evolution). I could not open either (publisher 403), so **which exact data produced `NEON.pt` is [?]**.
- A model trained mainly on NEON (US, airborne, 10 cm) is **not validated on other biomes, sensors or resolutions**. Treat accuracy on OpenAerialMap imagery as unknown until measured [?].

## 3. Preprocessing required by the paper / package

| Step | Rule | Source |
|---|---|---|
| Bands / order | 3-band **RGB** (DeepForest rejects anything else) | [V] `preprocess.py` |
| Source dtype | uint8 0-255 (`/255.0` -> float32 [0, 1]) | [V] `datasets/prediction.py` |
| Resolution | NEON RGB, **0.1 m/px** | [V] paper 2.1; docs say 400-800 px windows suit 0.1 m data |
| Window | **400x400 px**, 5 % overlap ("balance between memory constraints and providing the model sufficient spatial context") | [V] paper 2.3; `conf/config.yaml` |
| Model transform | ImageNet normalise, resize short side to 800, pad to 32 (section 2.2) | [V] |

Implementation (`td_hot.preprocess`): `read_rgb_chip` -> `sliding_windows` -> `preprocess_window` (PIL bilinear on normalised float planes; differs from torch's bilinear by < 0.05 normalised units on noisy test images, and only for down-scaling is PIL antialiasing a real difference) [V, tested]. `meta` stores, per window, `x, y, width, height, resized_width, resized_height`, plus chip `transform`/`crs`, which is everything needed to go back to chip pixels.

## 4. Model-space decoding (no CRS, no affine)

For each window `session.run` returns two raw tensors (ONNX names `cls_logits`, `bbox_regression`) [P]:

| Output | Shape | Meaning |
|---|---|---|
| `cls_logits` | `(1, A, 1)` | logits, all five FPN levels concatenated; `A = 9 * sum(HxW over levels)` (e.g. 120,087 for an 800x800 input) |
| `bbox_regression` | `(1, A, 4)` | `(dx, dy, dw, dh)` relative to the anchor; box weights `(1,1,1,1)` |

Decode (`td_hot.decode.decode_retinanet_output`, mirrors `RetinaNet.postprocess_detections`) [V, parity-tested]:

1. Anchors for the **padded** input: level grid sizes `H/8, H/16, H/32`, then `ceil(/2)` twice for P6, P7; per-level stride is the **integer division** `H // grid` (so P6 at 800 px has stride 61, not 64), which torchvision does and must be reproduced.
2. Per level: `score = sigmoid(logit)`, keep `score > confidence_threshold`, keep the top 1000 per level.
3. Box = standard delta decode, `dw`, `dh` clamped at `log(1000/16)`; clip to the **resized image** (not the padding).
4. NMS per class at `nms_threshold`, keep at most 300 per window.

Output: boxes in resized-window pixels. All of this is independent of georeferencing.

## 5. Geospatial postprocessing (map space)

1. Window -> chip pixels: `x_chip = x / scale_x + window.x` (same for y) with `scale = resized / original` [V].
2. **Cross-window NMS** in chip pixels at `iou_threshold` (DeepForest `predict.mosiac`). Without it, trees in the overlap strips are reported twice [V].
3. Clip to the chip, drop empty boxes, convert the 4 corners with the chip affine, reproject to EPSG:4326 with `pyproj` (`always_xy`) [P, same as sibling libs].
4. Difference from DeepForest: it casts merged coordinates to `int`; this library keeps floats [V]. No area or length filter is applied (it would need CRS units) [P].

## 6. Parameters

### 6.1 Inference (`predict_session(session, input_images, params)`)

All keys optional; `None`/missing -> default. Aliases are accepted, the canonical key wins.

| Key | Type | Default | Safe range | Stage | Purpose |
|---|---|---|---|---|---|
| `patch_size` | int | 400 | >= 32; keep near 400 px at 10 cm | window | window edge in **source** pixels (DeepForest `patch_size`) |
| `patch_overlap` | float | 0.05 | [0, 1) | window | fractional overlap between windows |
| `confidence_threshold` (alias `score_thresh`) | float | 0.1 | [0, 1] | decode | min sigmoid score. Package default 0.1; the 2019 paper filtered at 0.2 |
| `nms_threshold` (alias `nms_thresh`) | float | 0.05 | [0, 1] | decode | per-window NMS IoU (DeepForest `nms_thresh`) |
| `iou_threshold` | float | 0.15 | [0, 1] | map-space merge | cross-window NMS IoU (DeepForest `predict_tile` default; paper: boxes "overlapped by more than 15 %") |

Defaults are copied from DeepForest 2.1.0 `conf/config.yaml` and `main.predict_tile` [V]. DeepForest's own docstring text and defaults disagree between releases, so the values here come from the 2.1.0 code, not prose.

### 6.2 Split / training (reference; fine-tuning is **not implemented**, see section 7)

| Key | Type | Default | Purpose |
|---|---|---|---|
| `val_ratio` | float | 0.2 | fraction of chips for validation (fAIr-required) |
| `split_seed` | int | 42 | RNG seed (fAIr-required) |
| `block_size` | int | 4 | spatial block edge in OAM tile units for the split |

DeepForest's training defaults for later work: `lr 0.001`, `epochs 1`, `batch_size 1`, `HorizontalFlip(p=0.5)`; fine-tuning is a one-class head, so `num_classes=1` [V config]. Device: CUDA only when visible, else CPU; serving is CPU ONNX [P].

## 7. Sliding window and spatial split

### 7.1 Sliding window at inference

Why: the network sees fixed-size inputs and was trained on 400 px crops; a 1 km tile (10,000 px) cannot go through at once (paper: "the entire 1 km RGB tile cannot fit into GPU memory"). Windows overlap so a tree cut by one window is whole in a neighbour; duplicates are then removed by NMS.

Algorithm (identical to `slidingwindow.generateForSize`, which DeepForest uses) [V, tested]:

```
win      = min(patch_size, image_side)                     # a chip smaller than the patch is one window
step     = win - floor(win * patch_overlap)                # 400 -> 380
offsets  = range(0, side - win + 1, step) + [side - win]   # last window shifted back to the edge
```

Example, width 1000, patch 400, overlap 0.05: x offsets `0, 380, 600`, every window exactly 400 px, full coverage; the last window overlaps its neighbour by 180 px.

Rules that matter:
- Windows are cut on the **source** pixels, then resized by the model transform. They are not resampled to 10 cm.
- A 256 px fAIr chip is one window upscaled 3.125x. A 1 m/px chip is therefore a very different scale from the training data. Whether to resample so the ground resolution matches ~10 cm is **[?]** (section 8, Q1).
- DeepForest docs warn that predictions are sensitive to `patch_size` on new data and suggest tuning it per resolution.
- Merge happens in chip-pixel space before georeferencing, so it never needs CRS units.

### 7.2 Spatial split for training / validation

Neighbouring chips share canopy and trees cut at chip borders, and windows overlap by 5 %, so a random per-chip or per-window split leaks near-duplicates into validation and inflates metrics. `td_hot.dataset.spatial_split(chip_names, val_ratio, seed, block_size=4)`:

1. parse OAM tile names `OAM-<x>-<y>-<z>.ext`;
2. group chips into blocks `(x // block_size, y // block_size)`;
3. shuffle blocks with `numpy.default_rng(seed)`; fill validation with **whole blocks** until `val_ratio` is reached; the rest is train;
4. a name not matching the pattern raises `ValueError` (fAIr training chips are always OAM tiles).

Deterministic for a given seed, disjoint, full cover [V, tested]. This is a **design choice [P]**; the 2019 paper evaluates on separately hand-annotated images at a site and I did not verify its exact split procedure. Evaluating a fine-tuned model on windows of the same tile it trained on must be avoided for the same leakage reason.

## 8. Export, verification and open questions

### 8.1 ONNX export (`td_hot.export`, `train` extra) [V]

| Item | Value |
|---|---|
| Input | `images` `(1, 3, H, W)` float32; `H`, `W` dynamic (multiples of 32), already normalised and padded |
| Outputs | `cls_logits (1, A, 1)`, `bbox_regression (1, A, 4)`; `A` dynamic |
| Opset / exporter | 17, legacy exporter (`dynamo=False`), torch 2.14.1 |
| CPU runtime | onnxruntime CPU; no custom ops (NMS and decoding stay out of the graph) |

### 8.2 Checks actually run (this repo, 2026-10-09)

- 41 offline unit tests (`uv run pytest -q`): params, windows, preprocess, decode, serve, georeferencing, split.
- 9 opt-in integration tests (`uv run pytest -m integration`, need `weights/NEON.pt`): anchors equal torchvision's; preprocessing equals `GeneralizedRCNNTransform` (shape, resized size, values within 0.05); ONNX raw outputs equal torch (atol 2e-3); numpy decode equals `postprocess_detections` (same count, scores atol 2e-3, boxes atol 0.5 px) on 400x400, 256x256 and 300x420 inputs.
- **End-to-end vs real DeepForest 2.1.0** (`load_model("weecology/deepforest-tree")`, `predict_tile(patch_size=400, patch_overlap=0.05, iou_threshold=0.15)`) on its bundled NEON sample `OSBS_029.tif` tiled 3x2 into an 800x1200 image (12 windows): DeepForest 338 boxes, this library 337; 337/338 DeepForest boxes have a match with IoU > 0.8, 295/338 with IoU > 0.9, median best IoU 0.945. The residual gap is consistent with DeepForest's integer truncation of coordinates and PIL-vs-torch resize; I did not isolate each cause. One image only, and it is the training-domain sample.

### 8.3 Open questions

| # | Question | Smallest experiment |
|---|---|---|
| Q1 | OAM chips are rarely 10 cm. Does resampling to ~10 cm GSD before windowing beat running at native scale? | Run both on a few hand-labelled fAIr chips at 0.1-0.6 m and compare recall/precision at IoU 0.5 |
| Q2 | Score cut-off: package 0.1 vs paper 0.2? | Sweep `confidence_threshold` on labelled chips |
| Q3 | Exact training data of `NEON.pt` (cross-site paper) | Read the 2020 and MEE papers; both returned 403 here |
| Q4 | fAIr STAC / `pipeline.py` shapes for detection and parameter prefixes | Not written: the model pack is out of scope; check `fAIr-models/docs/contributing/model.md` |
| Q5 | Fine-tuning path (chips + GeoJSON -> boxes -> torchvision training -> ONNX) | Not implemented; needs a labelled fAIr dataset to validate |
| Q6 | Behaviour on dense canopy where crowns touch (box NMS 0.05 is very strict) | Count recall on a dense-forest chip while varying `nms_threshold` |

### 8.4 Sources

- Model card: https://huggingface.co/weecology/deepforest-tree (MIT, semi-supervised LiDAR + hand annotation, 32.3M params, files `NEON.pt`, `model.safetensors`, `config.json`).
- Weinstein et al., *Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks*, Remote Sensing 11(11):1309 (2019); preprint https://www.biorxiv.org/content/10.1101/532952 (ResNet-50 RetinaNet, 400x400 windows, 5 % overlap, NMS at 15 %, score filter 0.2, 0.1 m NEON RGB, recall 0.69 / precision 0.60).
- DeepForest 2.1.0 source (`deepforest/main.py`, `predict.py`, `preprocess.py`, `datasets/prediction.py`, `models/retinanet.py`, `conf/config.yaml`), docs https://deepforest.readthedocs.io; `slidingwindow` 0.0.14.
- Lin et al., *Focal Loss for Dense Object Detection*, https://arxiv.org/abs/1708.02002.
- torchvision `models/detection/retinanet.py`, `anchor_utils.py`, `transform.py` (torchvision 0.27/0.29).
