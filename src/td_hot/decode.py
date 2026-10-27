"""ONNX candidates -> final detections in model-input pixels (no CRS / affine dependence).

The level-2 graph already did anchors, score ranking, box decoding and clipping. What is left,
and stays tunable, is the confidence filter, NMS and the per-image cap (DeepForest's
`score_thresh`, `nms_thresh`, `detections_per_img`).
"""

from typing import Any

import numpy as np

from .params import DETECTIONS_PER_IMG, PostprocessParams


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS on XYXY boxes; returns kept indices in descending score order."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        overlap_w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        overlap_h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = overlap_w * overlap_h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode_detections(
    boxes: Any, scores: Any, labels: Any, cfg: PostprocessParams | None = None
) -> list[dict[str, float | int]]:
    """Mirror of the tail of torchvision `RetinaNet.postprocess_detections` for one image.

    Args:
        boxes: `(N, 4)` XYXY candidates in model-input pixels (graph output `boxes`).
        scores: `(N,)` sigmoid scores (`scores`).
        labels: `(N,)` class ids (`labels`).
    Returns:
        Detections as dicts `x1,y1,x2,y2,confidence,class_id`, best first, in model-input pixels.
    """
    cfg = cfg or PostprocessParams()
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    keep = scores > cfg.confidence_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    kept: list[int] = []
    for class_id in np.unique(labels):
        members = np.flatnonzero(labels == class_id)
        kept.extend(int(members[i]) for i in nms(boxes[members], scores[members], cfg.nms_threshold))
    kept.sort(key=lambda i: -scores[i])
    return [
        {
            "x1": float(boxes[i, 0]),
            "y1": float(boxes[i, 1]),
            "x2": float(boxes[i, 2]),
            "y2": float(boxes[i, 3]),
            "confidence": float(scores[i]),
            "class_id": int(labels[i]),
        }
        for i in kept[:DETECTIONS_PER_IMG]
    ]
