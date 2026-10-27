"""Evaluation helpers for td-hot: runs the shipped numpy decode on a torch model (`train` extra)."""

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import list_chip_paths, load_labels
from .decode import decode_detections
from .georef import detections_to_chip_pixels
from .params import parse_postprocess_params, parse_preprocess_params
from .preprocess import PreprocessMeta, preprocess_rgb, preprocess_chip
from .serve import detect_chip

log = logging.getLogger(__name__)


def _iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
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
        np.asarray([pred["x1"], pred["y1"], pred["x2"], pred["y2"]], dtype=np.float32)
        for pred in sorted(preds, key=lambda det: -float(det.get("confidence", 0.0)))
    ]
    gt_boxes = [np.asarray([gt["x1"], gt["y1"], gt["x2"], gt["y2"]], dtype=np.float32) for gt in gts]
    used = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if used[gt_index]:
                continue
            iou = _iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_iou >= float(iou_threshold) and best_index >= 0:
            used[best_index] = True
            tp += 1
        else:
            fp += 1
    fn = int(len(gt_boxes) - sum(used))
    return int(tp), int(fp), int(fn)


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
        paths = [path for path in paths if path.name in wanted]
    return pick_chips(paths, max_chips)


def _detect(model: Any, image: np.ndarray, meta: PreprocessMeta, params: dict[str, Any] | None):
    """One preprocessed image through the torch model's level-2 computation, decoded to chip pixels."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install td-hot with `[train]` extras to evaluate torch models.") from exc
    from .export import candidate_detector

    model.eval()
    device = next(model.parameters()).device  # runs wherever the model lives (CPU or GPU)
    with torch.no_grad():
        boxes, scores, labels = candidate_detector(model)(torch.from_numpy(image).to(device))
    dets = decode_detections(
        boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(), parse_postprocess_params(params)
    )
    return detections_to_chip_pixels(dets, meta)


def detect_chip_torch(
    model: Any, image_path: str | Path, params: dict[str, Any] | None
) -> list[dict[str, float | int]]:
    """Torch model -> the same level-2 computation as the ONNX -> `decode_detections` -> chip pixels.

    Requires the `train` extra at runtime. The model may be on CPU or GPU.
    """
    image, meta = preprocess_chip(image_path, parse_preprocess_params(params))
    return _detect(model, image, meta, params)


def _score(paths, label_map, detect, iou_thresholds) -> dict[float, dict[str, float]]:
    """P/R/F1 over `paths` at each IoU, one `detect(path)` call per chip; logs one INFO line per threshold."""
    totals = {iou: [0, 0, 0] for iou in iou_thresholds}
    started = time.time()
    for path in paths:
        preds = detect(path)
        for iou, tally in totals.items():
            for idx, value in enumerate(
                match_detections(preds, label_map.get(path.name, []), iou_threshold=iou)
            ):
                tally[idx] += value
    results = {iou: _prf(*tally) for iou, tally in totals.items()}
    for iou, metrics in results.items():
        log.info(
            "evaluation on %d chips, IoU>=%.2f: precision=%.3f recall=%.3f f1=%.3f (tp=%d fp=%d fn=%d) in %.0fs",
            len(paths), iou, metrics["precision"], metrics["recall"], metrics["f1"], metrics["tp"], metrics["fp"],
            metrics["fn"], time.time() - started,
        )  # fmt: skip
    return results


def score_chip(
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
    image, meta = preprocess_rgb(rgb, scale)
    preds = _detect(model, image, meta, params)
    return _prf(*match_detections(preds, truth, iou_threshold=iou_threshold))


def score_dataset(
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
    label_map = load_labels(labels_json)
    paths = _chips(images_dir, names, max_chips)
    return _score(paths, label_map, lambda path: detect_chip_torch(model, path, params), iou_thresholds)


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
    label_map = load_labels(labels)
    paths = _chips(images_dir, names)
    return _score(paths, label_map, lambda path: detect_chip(session, path, params)[0], iou_thresholds)


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
    label_map = load_labels(labels_json)
    low_params = {**(params or {}), "confidence_threshold": min(thresholds)}
    totals = {threshold: [0, 0, 0, 0] for threshold in thresholds}  # tp, fp, fn, detections
    for path in _chips(images_dir, names, max_chips):
        preds = detect_chip_torch(model, path, low_params)
        for threshold, tally in totals.items():
            kept = [det for det in preds if det["confidence"] > threshold]
            for idx, value in enumerate(
                match_detections(kept, label_map.get(path.name, []), iou_threshold=iou)
            ):
                tally[idx] += value
            tally[3] += len(kept)
    rows = [
        {"threshold": threshold, "detections": float(tally[3]), **_prf(tally[0], tally[1], tally[2])}
        for threshold, tally in totals.items()
    ]
    best = max(rows, key=lambda row: row["f1"])
    log.info(
        "confidence sweep at IoU>=%.2f: best f1=%.3f at confidence %.2f", iou, best["f1"], best["threshold"]
    )
    return rows
