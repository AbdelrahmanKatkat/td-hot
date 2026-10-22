"""Evaluation helpers for td-hot: runs the shipped numpy decode on a torch model (`train` extra)."""

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import list_chip_paths
from .decode import decode_detections
from .georef import detections_to_chip
from .params import parse_postprocess_params, parse_preprocess_params
from .preprocess import PreprocessMeta, preprocess_array, preprocess_chip
from .serve import detect_chip

log = logging.getLogger(__name__)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / (area_a + area_b - inter + 1e-9))


def match_detections(
    preds: list[dict[str, float | int]],
    gts: list[dict[str, float | int]],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy matching by confidence: returns (tp, fp, fn)."""
    if not preds and not gts:
        return 0, 0, 0
    pred_boxes = [
        np.asarray([p["x1"], p["y1"], p["x2"], p["y2"]], dtype=np.float32)
        for p in sorted(preds, key=lambda d: -float(d.get("confidence", 0.0)))
    ]
    gt_boxes = [np.asarray([g["x1"], g["y1"], g["x2"], g["y2"]], dtype=np.float32) for g in gts]
    used = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    for pb in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if used[j]:
                continue
            iou = _iou_xyxy(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= float(iou_threshold) and best_j >= 0:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = int(len(gt_boxes) - sum(used))
    return int(tp), int(fp), int(fn)


def predict_chip_torch(
    model: Any, image_path: str | Path, params: dict[str, Any] | None
) -> list[dict[str, float | int]]:
    """Torch model -> the same level-2 computation as the ONNX -> `decode_detections` -> chip pixels.

    Requires the `train` extra at runtime. The model may be on CPU or GPU.
    """
    image, meta = preprocess_chip(image_path, parse_preprocess_params(params))
    return _detect(model, image, meta, params)


def _detect(model: Any, image: np.ndarray, meta: PreprocessMeta, params: dict[str, Any] | None):
    """One preprocessed image through the torch model's level-2 computation, decoded to chip pixels."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install td-hot with `[train]` extras to evaluate torch models.") from exc
    from .export import level2_network

    model.eval()
    device = next(model.parameters()).device  # runs wherever the model lives (CPU or GPU)
    with torch.no_grad():
        boxes, scores, labels = level2_network(model)(torch.from_numpy(image).to(device))
    dets = decode_detections(
        boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(), parse_postprocess_params(params)
    )
    return detections_to_chip(dets, meta)


def score_image(
    model: Any,
    rgb: np.ndarray,
    truth: list[dict[str, float | int]],
    *,
    scale: float,
    params: dict[str, Any] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Precision / recall / F1 on one RGB image at an explicit scale, for example a source-domain reference.

    Fine-tuning can make a model forget the domain its checkpoint came from. Scoring the base model and the
    fine-tuned model on the same reference image (with its own scale, not the zoom rule) measures that.
    """
    image, meta = preprocess_array(rgb, scale)
    preds = _detect(model, image, meta, params)
    return _prf(*match_detections(preds, truth, iou_threshold=iou_threshold))


def pick_chips(paths: list[Path], max_chips: int | None) -> list[Path]:
    """At most `max_chips` chips, evenly spaced through the sorted list (so a pilot still spans the area)."""
    if not max_chips or max_chips >= len(paths):
        return paths
    step = len(paths) / max_chips
    return [paths[int(i * step)] for i in range(max_chips)]


def _chips(images_dir: str | Path, names: Sequence[str] | None, max_chips: int | None = None) -> list[Path]:
    paths = list_chip_paths(images_dir)
    if names is not None:
        wanted = set(names)
        paths = [p for p in paths if p.name in wanted]
    return pick_chips(paths, max_chips)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _load_labels(
    labels: str | Path | Mapping[str, list[dict[str, Any]]],
) -> Mapping[str, list[dict[str, Any]]]:
    if isinstance(labels, (str, Path)):
        labels = json.loads(Path(labels).read_text(encoding="utf-8"))
    if not isinstance(labels, Mapping):
        raise ValueError("labels must map chip filename -> list of boxes")
    return labels


def _score(paths, label_map, detect, iou_thresholds) -> dict[float, dict[str, float]]:
    """P/R/F1 over `paths` at each IoU, one `detect(path)` call per chip; logs one INFO line per threshold."""
    counts = {iou: [0, 0, 0] for iou in iou_thresholds}
    started = time.time()
    for path in paths:
        preds = detect(path)
        for iou, c in counts.items():
            for i, v in enumerate(match_detections(preds, label_map.get(path.name, []), iou_threshold=iou)):
                c[i] += v
    out = {iou: _prf(*c) for iou, c in counts.items()}
    for iou, m in out.items():
        log.info(
            "evaluation on %d chips, IoU>=%.2f: precision=%.3f recall=%.3f f1=%.3f (tp=%d fp=%d fn=%d) in %.0fs",
            len(paths), iou, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"], time.time() - started,
        )  # fmt: skip
    return out


def score_chips(
    model: Any,
    *,
    images_dir: str | Path,
    labels_json: str | Path | Mapping[str, list[dict[str, Any]]],
    params: dict[str, Any] | None = None,
    names: Sequence[str] | None = None,
    iou_thresholds: tuple[float, ...] = (0.5,),
    max_chips: int | None = None,
) -> dict[float, dict[str, float]]:
    """Precision / recall / F1 of a torch `model` on labelled chips, at each IoU threshold (one pass per chip).

    `labels_json` is a `labels.json` path or the mapping itself. A chip missing from it counts as having no trees.
    `names` limits the scoring to those chips (for example a validation split of one chips folder).
    """
    label_map = _load_labels(labels_json)
    paths = _chips(images_dir, names, max_chips)
    return _score(paths, label_map, lambda p: predict_chip_torch(model, p, params), iou_thresholds)


def score_session(
    session: Any,
    *,
    images_dir: str | Path,
    labels: str | Path | Mapping[str, list[dict[str, Any]]],
    names: Sequence[str] | None = None,
    params: dict[str, Any] | None = None,
    iou_thresholds: tuple[float, ...] = (0.5,),
) -> dict[float, dict[str, float]]:
    """Precision / recall / F1 through an ONNX session: the deployed path, with no torch involved.

    `names` limits the scoring to those chips (for example a validation split). This is what a pipeline
    step should report as the metrics of a model it just exported.
    """
    label_map = _load_labels(labels)
    paths = _chips(images_dir, names)
    return _score(paths, label_map, lambda p: detect_chip(session, p, params)[0], iou_thresholds)


def sweep_confidence(
    model: Any,
    *,
    images_dir: str | Path,
    labels_json: str | Path | Mapping[str, list[dict[str, Any]]],
    names: Sequence[str] | None = None,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    iou: float = 0.5,
    params: dict[str, Any] | None = None,
    max_chips: int | None = None,
) -> list[dict[str, float]]:
    """Precision / recall / F1 at each confidence threshold, from one prediction per chip.

    Predicts at the lowest threshold and filters upward. That is exact: greedy NMS never lets a lower-scoring
    box suppress a higher-scoring one, and the per-chip cap keeps the best scores either way. Use it to choose
    the `confidence_threshold` of a freshly trained model, whose scores are not on the scale of the base model's.
    """
    label_map = _load_labels(labels_json)
    floor = {**(params or {}), "confidence_threshold": min(thresholds)}
    counts = {t: [0, 0, 0, 0] for t in thresholds}  # tp, fp, fn, detections
    for path in _chips(images_dir, names, max_chips):
        preds = predict_chip_torch(model, path, floor)
        for t, c in counts.items():
            kept = [d for d in preds if d["confidence"] > t]
            for i, v in enumerate(match_detections(kept, label_map.get(path.name, []), iou_threshold=iou)):
                c[i] += v
            c[3] += len(kept)
    rows = [{"threshold": t, "detections": float(c[3]), **_prf(c[0], c[1], c[2])} for t, c in counts.items()]
    best = max(rows, key=lambda r: r["f1"])
    log.info(
        "confidence sweep at IoU>=%.2f: best f1=%.3f at confidence %.2f", iou, best["f1"], best["threshold"]
    )
    return rows
