# Theory: DeepForest tree-crown detector (`weecology/deepforest-tree`)

Contract for the `td-hot` library. Tags: **[V]** verified from source code, the checkpoint, or a run in this repo;
**[P]** proposed design decision; **[?]** unknown / needs an experiment (section 8).

## 0. Summary

| Item | Value |
|---|---|
| Task / geometry | Object detection, one class `Tree` (id 0). Output is an axis-aligned **box polygon** per tree crown [V] |
| Model | torchvision **RetinaNet, ResNet-50 + FPN**, 1 class, 32,254,189 parameters [V] |
| Input | 3-band RGB. The checkpoint was trained on **10 cm** imagery cut into 400 px windows and upscaled 2x to 800 px, i.e. **5 cm of ground per model pixel** [V: paper, DeepForest config] |
| Inference (upstream) | Sliding window (400 px, 5 % overlap) -> RetinaNet -> NMS per window -> NMS across windows [V]. **This repo drops the windowing**: fAIr chips are 256 px, so one chip is one model pass [P] |
| This repo | One scale step from the chip's **zoom level** (section 3), then an ONNX graph that contains normalisation, the network, anchors and box decoding ("level 2", section 4); confidence filter, NMS and georeferencing are Python [P] |
| Weights | HF `weecology/deepforest-tree` (MIT per the model card); `weights/NEON.pt` is the same network as a plain state dict [V] |

## 1. Task and output contract

- Task `object-detection`, geometry `polygon` (each box becomes a closed 5-point ring, EPSG:4326). Crowns are boxes, not outlines: the model never predicts a crown shape [V].
- GeoJSON `FeatureCollection`; per-feature `properties`:

| Property | Type | Meaning |
|---|---|---|
| `confidence` | float, 4 dp | sigmoid score of the RetinaNet classification logit, above the confidence threshold |
| `class_id` | int | always `0` |
| `class_name` | str | `"Tree"` (HF `config.json` `label_dict = {"Tree": 0}`) |
| `source` | str | chip file name |

- Empty prediction -> `{"type": "FeatureCollection", "features": []}` [V, tested].
- Chips without a CRS/transform (plain PNG/JPEG) produce polygons in **pixel coordinates**, not lon/lat. Their *scale* is still correct because it comes from the file name (section 3). fAIr chips are GeoTIFFs, so this only affects local experiments [V].
- Chip files must be named `OAM-<x>-<y>-<z>.<ext>`; any other name raises `ValueError` [P].

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
- **The checkpoint stores weights only.** Normalisation constants, anchors, thresholds and the resize rule are *not* in the `.pt` (checked: the only keys containing `mean`/`std` are BatchNorm running statistics). They are set by `td_hot.export.build_retinanet`, which training, evaluation and export all use, so every generation of a model (base, fine-tuned, re-fine-tuned) gets the same constants [V].
- **Anchors** (torchvision default, used by DeepForest): per level sizes `(32,40,50) (64,80,101) (128,161,203) (256,322,406) (512,645,812)` x aspect ratios `(0.5, 1, 2)` = 9 per location. This library no longer re-implements them: torchvision's own anchor and decode code is traced into the ONNX graph [V].
- **Loss** (training only): focal loss for classification and L1 for box regression, as in RetinaNet [Lin et al. 2017] [V: paper cites it as the detector; loss is torchvision's].
- ResNet-101/152 were tried in the paper and gave no gain, so ResNet-50 is final [V: paper 2.3].

### 2.2 The torchvision transform and what this library does with it [V]

torchvision's `GeneralizedRCNNTransform(min_size=800, max_size=1333, ImageNet mean/std)` does, in order: normalise, resize (short side to 800, long side capped at 1333), pad to a multiple of 32. DeepForest does not override it, so 10 cm / 400 px windows were upscaled 2x.

In this library:
1. **Normalisation** `(x - mean) / std`, mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`: inside the ONNX graph.
2. **Padding** to a multiple of 32: inside the ONNX graph.
3. **Resize**: replaced by `preprocess.py`, with the scale taken from the chip's zoom (section 3) instead of "short side = 800". torchvision's resize is disabled in `build_retinanet`, so training and serving see identical pixels. torchvision's `max_size = 1333` is replaced by `max_input_px` (default 1024, section 6.1): the longest side of the scaled chip never exceeds it, so a coarse zoom is upscaled less instead of growing without bound. A 256 px chip at zoom 18 (latitude 28) would otherwise become about 2,700 px and need about 3.8 GiB for one CPU forward pass.

## 3. Preprocessing: scale from the zoom level

### 3.1 Why scale matters [V: measured, section 8.2]

The network recognises crowns at a particular pixel size. The training scale is **0.05 m of ground per model pixel** (10 cm x 400 px / 800 px). Feeding a chip whose crowns are 3x smaller finds almost nothing. On the NEON sample (61 hand-annotated trees, F1 at IoU 0.5, conf >= 0.2), running genuine 256 px chips downsampled to a coarser resolution:

| Chip resolution | Fixed 3.125x upscale | Scale = chip GSD / 0.05 m |
|---|---|---|
| 10 cm | 0.15 | **0.61** |
| 12.5 cm | 0.42 | **0.66** |
| 15.6 cm | 0.47 | 0.49 |
| no upscale at 15.6 cm | 0.00 | n/a |

One image, in the model's training domain, with resolution emulated by area-downsampling; real coarser imagery has less detail, so these are optimistic.

### 3.2 The rule

```
chip_gsd_m = 40075016.68558 * cos(lat_of_tile_centre) / 2^z / chip_width_px     # true ground size of a chip pixel
scale      = chip_gsd_m / model_gsd_m                                              # model_gsd_m = 0.05
scale      = min(scale, max_input_px / longest chip side)                         # max_input_px = 1024
input size = round(chip size * scale)                                              # one bilinear resize, in float
```

`z`, and `y` (which gives the latitude) come from the chip name `OAM-<x>-<y>-<z>.<ext>` (`td_hot.tiles`). Nothing is read from the file, so PNG/JPEG chips work the same as GeoTIFFs. Examples for a 256 px chip: zoom 20 at the equator is 14.9 cm/px -> scale 2.99; the same zoom at 48 deg latitude is 10 cm/px -> scale 2.0. Ground resolution depends on `cos(latitude)`, so the latitude matters as much as the zoom; XYZ and TMS row numbering give the same value because `cos` is symmetric about the equator [V: formula, unit-tested].

`PreprocessParams.model_gsd_m` (default 0.05, range [0.005, 1]) sets the scale and `max_input_px` (default 1024, range [64, 8192]) caps it: zoom 20 is untouched (about 760 px), zoom 19 goes from 5.9 to 4.0 at Freetown, zoom 18 from 10.6 to 4.0. **Training and serving must use the same value**; `train_retinanet` takes the same `PreprocessParams` and scales chips and boxes with it [P]. Scales above 16 (a 256 px chip coarser than 80 cm/px) raise `ValueError`; behaviour between 15.6 and 40 cm is not measured [?].

### 3.3 Is the zoom always known? [V: checked, with limits]

- **OpenAerialMap itself has no zoom.** An OAM image is a GeoTIFF with a native resolution (`gsd`: median 5 cm across 60 recent images, range 0.9 cm to 38 m) and a TMS endpoint `.../{z}/{x}/{y}` that serves any zoom. Zoom exists only in the tile request.
- **fAIr chips carry it:** sibling libraries and fAIr-models samples name chips `OAM-{x}-{y}-{z}.tif` and parse exactly this pattern (`pl-hot`, `spd-hot`). I could not open fAIr's own source here, so the claim "z is the third field" rests on those libraries, not on fAIr documentation [?].
- Zoom gives the *nominal* resolution. If the source imagery is coarser than the zoom, tiles are upsampled and look blurry: the scale is then right but the detail is not.

### 3.4 Normalisation on OAM imagery [V: sampled, small sample]

24 recent OAM images (9 tiles each, at the zoom nearest 15 cm; opaque pixels only, RGB in [0, 1]; several rows are the same image), against ImageNet and the NEON sample:

| | mean R, G, B | std R, G, B |
|---|---|---|
| ImageNet (used by the model) | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
| NEON sample `OSBS_029` | 0.614, 0.630, 0.537 | 0.200, 0.191, 0.160 |
| OAM, average over images | 0.547, 0.521, 0.430 | 0.180, 0.159, 0.171 |
| OAM, spread of per-image means | 0.40-0.66, 0.41-0.60, 0.33-0.51 | |

Decision [P]: **keep the ImageNet constants fixed.** OAM is slightly brighter and lower-contrast than ImageNet, but it varies between images (std of per-image means about 0.06-0.08) by more than it differs from ImageNet, and the NEON imagery the model was trained on is further from ImageNet than OAM is. There is no single "OAM value". Fine-tuning on OAM adapts the weights to whatever the input looks like; what breaks a model is training and serving with *different* constants, which cannot happen here because the constants are written into every exported graph. Dataset-specific statistics are untested and would have to be carried with the model.

## 4. The ONNX contract (level 2)

| | |
|---|---|
| Input `image` | `(3, H, W)` float32, RGB, values in [0, 1], `H` and `W` dynamic. One chip per call, no batch dimension |
| Output `boxes` | `(N, 4)` XYXY in input pixels, clipped to the input, best score first |
| Output `scores` | `(N,)` sigmoid scores |
| Output `labels` | `(N,)` int64 class ids (always 0) |

**Inside the graph** (torchvision code, traced; `td_hot.export.level2_network`): ImageNet normalisation, padding to 32, ResNet-50 + FPN + heads, anchors for the padded input, sigmoid, the top 1000 candidates per FPN level, box decoding (`dw`, `dh` clamped at `log(1000/16)`), clipping to the input, and a sort by score. N is at most 5 x 1000.

**Outside, tunable** (`td_hot.decode.decode_detections`): keep `score > confidence_threshold`; NMS per class at `nms_threshold`; keep at most 300 [V]. The graph's own score floor is 0 instead of DeepForest's 0.1, which gives identical results once the filter is applied outside, because the top-1000 selection keeps the highest scores either way [V: parity-tested at thresholds 0.02 and 0.1].

**Not in the graph, by design:** georeferencing and reprojection (they need the chip's transform and CRS), and NMS (so `nms_threshold` and `confidence_threshold` stay tunable and the STAC value 0.38 works).

The ONNX asset in the shared STAC item (`kshitijrajsharma/fair-deepforest-tree-crowns`) is a different contract: fixed `[3,256,256]` input, resize, NMS and a 0.1 floor inside the graph. **This library does not accept it**; the base model needs an ONNX exported with `td_hot.export` [V: inspected].

## 5. Geospatial postprocessing (map space)

1. Model-input pixels -> chip pixels: `x_chip = x / scale_x` (same for y), `scale = resized / original` [V].
2. Upstream DeepForest runs a **cross-window NMS** (`iou_threshold` 0.15, `predict.mosiac`) because overlapping windows report a tree twice. With one pass per chip the per-chip NMS at 0.05 is already stricter, so a second pass could not remove anything; it is not implemented [P].
3. Clip to the chip, drop empty boxes, convert the 4 corners with the chip affine, reproject to EPSG:4326 with `pyproj` (`always_xy`) [P, same as sibling libs].
4. Difference from DeepForest: it casts merged coordinates to `int`; this library keeps floats [V]. No area or length filter is applied (it would need CRS units) [P].
5. Crowns cut by a chip border become two partial boxes, one per chip. OAM tiles do not overlap, so no NMS can merge them; this is a limitation of tiling [?].

## 6. Parameters (`td_hot.params`)

### 6.1 Inference (`predict_session(session, input_images, params)`)

All keys optional; `None`/missing -> default. Aliases are accepted, the canonical key wins. Unknown keys are ignored.

| Key | Type | Default | Safe range | Stage | Purpose |
|---|---|---|---|---|---|
| `model_gsd_m` | float | 0.05 | [0.005, 1] | preprocess | metres of ground per model pixel; sets the scale (section 3). Must match training |
| `max_input_px` | int | 1024 | [64, 8192] | preprocess | cap on the longest side of the scaled chip (section 3); bounds memory at coarse zooms. Must match training |
| `confidence_threshold` (alias `score_thresh`) | float | 0.1 | [0, 1] | decode | minimum sigmoid score; STAC default 0.38; the 2019 paper filtered at 0.2 |
| `nms_threshold` (alias `nms_thresh`) | float | 0.05 | [0, 1] | decode | NMS IoU over all candidates of the chip (DeepForest `nms_thresh`; no rationale in its `config.yaml`) |

Constants, not parameters: `IMAGE_MEAN`, `IMAGE_STD`, `TOPK_CANDIDATES = 1000` (fixed in the graph), `DETECTIONS_PER_IMG = 300`, `MAX_SCALE = 16`. Defaults come from DeepForest 2.1.0 `conf/config.yaml` [V].

### 6.2 Split / training (`td_hot.params.TrainParams`)

| Key | Type | Default | Purpose |
|---|---|---|---|
| `val_ratio` | float | 0.2 | fraction of chips for validation (fAIr-required) |
| `split_seed` | int | 42 | RNG seed (fAIr-required) |
| `block_size` | int | 4 | spatial block edge in OAM tile units for the split |
| `epochs`, `batch_size` | int | 5, 2 | |
| `learning_rate` (alias `lr`), `weight_decay` | float | 1e-3, 1e-4 | SGD with momentum 0.9 |
| `scheduler` | `cosine` / `none` | `cosine` | cosine decay to 0 over all batches of the run, stepped per batch (not per epoch) |
| `warmup_steps` | int | 0 | linear learning-rate ramp over the first batches |
| `max_grad_norm` | float | 1.0 | gradient clipping (0 disables) |
| `freeze_encoder` | bool | **true** | the ResNet encoder does not train; the FPN and heads do. BatchNorm is always frozen |
| `unfreeze_from` | `none`, `layer4`..`layer1` | `none` | with a frozen encoder, still train the stages from this one on (needs `freeze_encoder=true`) |
| `encoder_lr_factor` | float in (0, 1] | 1.0 | learning-rate multiplier for the encoder weights that do train |
| `early_stop_patience` | int | 5 | epochs without a better loss before stopping |
| `sample_fraction` | float | 1.0 | one fixed random subset of the training tiles for the whole run |
| `epoch_fraction` | float | 1.0 | each epoch trains on a fresh random fraction of the tiles; with highly correlated neighbouring tiles a short epoch gives more validation points per pass of data |
| `device`, `seed` | | `auto`, 42 | `cuda` if visible, RNG |

The defaults are the recipe of the fAIr `deepforest_tree` pack (SGD, cosine, clipping, flip augmentation, frozen encoder). DeepForest's own defaults for comparison: `lr 0.001`, `epochs 1`, `batch_size 1`, `HorizontalFlip(p=0.5)`, `num_classes=1` [V config]. Training chips are scaled with the same rule as serving.

**Why the encoder is frozen by default [M]:** on a few hundred Freetown tiles, training every weight kept only 32% to 66% of the base model's F1 on its own NEON domain, a frozen encoder kept 96% (and reached target F1 0.353 against 0.480 for the unfrozen run). `unfreeze_from` / `encoder_lr_factor` sit between the two; their measured effect is in `research/datasets.md` section 6.2.

### 6.3 Evaluation entry points (`td_hot.evaluate`)

`score_chips` (torch model) and `score_session` (ONNX session, the deployed path, no torch) give precision / recall / F1 at one or several IoU thresholds from one pass per chip; `sweep_confidence` picks the confidence threshold of a fine-tuned model; `score_image` scores one image at an explicit scale (the forgetting check on the base model's own domain). Labels are a `labels.json` path or a mapping (`dataset.labels_for_chips`), and `names` / `max_chips` select chips.

## 7. Windowing (upstream, not used here) and spatial split

### 7.1 Sliding window at inference

**Not implemented in this library.** fAIr chips are always 256 px, below the 400 px window, so every chip is already a single window (`win = min(400, 256)`) and the crop, overlap and cross-window merge would be dead code. For reference, upstream DeepForest cuts large mosaics because the network sees fixed-size inputs and a 1 km tile (10,000 px) cannot go through at once (paper: "the entire 1 km RGB tile cannot fit into GPU memory"); windows overlap by 5 % so a cut tree is whole in a neighbour, and duplicates are removed by NMS. The algorithm is `slidingwindow.generateForSize`: `win = min(patch, side)`, `step = win - floor(win * overlap)`, offsets `range(0, side - win + 1, step)` plus a last window shifted back to the edge.

Related facts [V]:
- A 256 px chip is rescaled in **one** resize; pre-resizing to 400 px first would only add a second interpolation.
- A 256 px chip at 15.6 cm covers 40 m, the same ground footprint as a 400 px x 10 cm training window; at that resolution the scale is 3.125 and matches training exactly.
- Web-Mercator resolution: zoom 20 gives 14.9 cm/px at the equator, 12.9 cm at 30 deg, 11.4 cm at 40 deg; zoom 19 gives 29.8, 25.8 and 22.9 cm. Zoom 20 is therefore the level to recommend in the STAC item (`fair:preview.zoom.recommended`); the schema does not say whether fAIr uses it when fetching chips [?].

### 7.2 Spatial split for training / validation

Neighbouring chips share canopy and trees cut at chip borders, so a random per-chip split leaks near-duplicates into validation and inflates metrics. `td_hot.dataset.spatial_split(chip_names, val_ratio, seed, block_size=4)`:

1. parse OAM tile names `OAM-<x>-<y>-<z>.ext`;
2. group chips into blocks `(x // block_size, y // block_size)`;
3. shuffle blocks with `numpy.default_rng(seed)`; fill validation with **whole blocks** until `val_ratio` is reached; the rest is train;
4. a name not matching the pattern raises `ValueError` (fAIr training chips are always OAM tiles).

Deterministic for a given seed, disjoint, full cover [V, tested]. This is a **design choice [P]**; the shared STAC item currently declares `fair:split_spec.strategy = "random"`, which does not describe it. A re-fine-tune on a new dataset may put earlier validation chips into training, so metrics are only comparable across generations if the held-out set is fixed [P].

## 8. Verification and open questions

### 8.1 ONNX export (`td_hot.export`, `train` extra) [V]

| Item | Value |
|---|---|
| Entry points | `build_retinanet(weights)`, `level2_network(model)`, `export_onnx_bytes(model)`, CLI `python -m td_hot.export NEON.pt out.onnx` |
| Input / outputs | section 4 |
| Opset / exporter | 17, legacy exporter (`dynamo=False`), torch 2.14.1, output validated with `onnx.checker` |
| Graph | contains `TopK`, no `NonMaxSuppression` (tested) |
| CPU runtime | onnxruntime CPU; serving needs no torch |

### 8.2 Checks actually run (this repo, 2026-10)

- 57 offline unit tests (`uv run pytest -q`): params, tiles and scale, preprocess, decode, serve (including a planted box mapped to lon/lat), georeferencing, split, label prep, train params, evaluation matching.
- 11 opt-in integration tests (`uv run pytest -m integration`, need `weights/NEON.pt`): the ONNX candidates equal the torch level-2 candidates as a set (scores atol 2e-3, boxes within 0.5 px) on 764x764, 800x800 and 600x840 inputs; `decode_detections` on the ONNX outputs equals torchvision's own detections (same count, scores atol 2e-3, boxes atol 0.5 px) at thresholds 0.02 and 0.1; the graph has no NMS op; a train + evaluate smoke test on random weights.
- **Real NEON image, level-2 ONNX in onnxruntime** (`OSBS_029.tif`, 61 annotated trees, chips named for zoom 20 at latitudes giving 10, 12.5 and 14.9 cm): F1 0.60, 0.65, 0.50 (section 3.1). Also, earlier on four real 256 px crops, the third-party full-graph ONNX, a full-graph export from this repo and the previous numpy pipeline agreed to 1e-4 px (same 45, 38, 43 and 42 boxes).
- **Earlier end-to-end vs DeepForest 2.1.0** (with the windowed numpy implementation, since removed): on the same image tiled to 800x1200, DeepForest 338 boxes, this library 337; 337/338 matched at IoU > 0.8. One image, training domain.

### 8.3 Open questions

| # | Question | Smallest experiment |
|---|---|---|
| Q1 | How accurate is the zoom-derived scale on real OAM imagery, and between 15.6 and 40 cm/px? | Hand-label a few OAM chips at zooms 19-21 and compare F1 at IoU 0.4/0.5 |
| Q2 | Score cut-off: package 0.1, paper 0.2, STAC 0.38 | Sweep `confidence_threshold` on labelled chips |
| Q3 | Exact training data of `NEON.pt` (cross-site paper) | Read the 2020 and MEE papers; both returned 403 here |
| Q4 | Does fAIr read `fair:preview.zoom.recommended` when fetching chips? | Check fAIr-models docs / code |
| Q5 | Fine-tuning on OAM labels | Needs a labelled fAIr dataset; only a random-weight smoke test exists |
| Q6 | Dense canopy where crowns touch (NMS 0.05 is very strict) | Count recall on a dense-forest chip while varying `nms_threshold` |
| Q7 | Would dataset-specific normalisation help after fine-tuning? | Fine-tune twice (ImageNet vs OAM statistics) and compare validation F1 |

### 8.4 Sources

- Model card: https://huggingface.co/weecology/deepforest-tree (MIT, semi-supervised LiDAR + hand annotation, 32.3M params, files `NEON.pt`, `model.safetensors`, `config.json`).
- Weinstein et al., *Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks*, Remote Sensing 11(11):1309 (2019); preprint https://www.biorxiv.org/content/10.1101/532952 (ResNet-50 RetinaNet, 400x400 windows, 5 % overlap, NMS at 15 %, score filter 0.2, 0.1 m NEON RGB, recall 0.69 / precision 0.60).
- DeepForest 2.1.0 source (`deepforest/main.py`, `predict.py`, `preprocess.py`, `datasets/prediction.py`, `models/retinanet.py`, `conf/config.yaml`), docs https://deepforest.readthedocs.io; `slidingwindow` 0.0.14.
- OpenAerialMap API https://api.openaerialmap.org/meta (`gsd`, TMS `{z}/{x}/{y}` URL per image).
- Lin et al., *Focal Loss for Dense Object Detection*, https://arxiv.org/abs/1708.02002.
- torchvision `models/detection/retinanet.py`, `anchor_utils.py`, `transform.py` (torchvision 0.29).
