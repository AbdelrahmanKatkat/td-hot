"""Evaluation helpers for td-hot: runs the shipped numpy decode on a torch model (`train` extra)."""

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import list_chip_paths
from .decode import decode_retinanet_output, merge_window_detections
from .georef import window_to_chip
from .params import parse_postprocess_params, parse_preprocess_params
from .preprocess import preprocess_chip


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
    """Torch model -> numpy decode pipeline -> merged chip-pixel detections.

    Requires the `train` extra at runtime.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install td-hot with `[train]` extras to evaluate torch models.") from exc

    pre_cfg = parse_preprocess_params(params)
    post_cfg = parse_postprocess_params(params)
    batches, meta = preprocess_chip(image_path, pre_cfg)

    model.eval()
    chip_dets: list[dict[str, float | int]] = []
    with torch.no_grad():
        for batch, window in zip(batches, meta.windows):
            t = torch.from_numpy(np.asarray(batch))
            feats = list(model.backbone(t).values())
            head = model.head(feats)
            cls_logits = head["cls_logits"].detach().cpu().numpy()
            bbox_reg = head["bbox_regression"].detach().cpu().numpy()
            dets = decode_retinanet_output(
                cls_logits,
                bbox_reg,
                input_hw=(int(batch.shape[2]), int(batch.shape[3])),
                image_hw=(int(window.resized_height), int(window.resized_width)),
                cfg=post_cfg,
            )
            for det in dets:
                x1, y1, x2, y2 = window_to_chip(det, window)
                chip_dets.append({**det, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return merge_window_detections(chip_dets, post_cfg.iou_threshold)


def evaluate_retinanet(
    model: Any,
    *,
    images_dir: str | Path,
    labels_json: str | Path,
    params: dict[str, Any] | None = None,
    iou_match_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute simple dataset-level precision/recall/F1 on chip-pixel boxes."""
    import json

    images_dir = Path(images_dir)
    label_map = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    if not isinstance(label_map, dict):
        raise ValueError("labels_json must be a dict mapping filename -> list of boxes")

    tp = fp = fn = 0
    for img_path in list_chip_paths(images_dir):
        preds = predict_chip_torch(model, img_path, params)
        gts = label_map.get(img_path.name, [])
        t, f_p, f_n = match_detections(preds, gts, iou_threshold=float(iou_match_threshold))
        tp += t
        fp += f_p
        fn += f_n

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
