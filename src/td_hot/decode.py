"""RetinaNet ONNX output -> boxes in model-input pixels (no CRS / affine dependence)."""

import math
from typing import Any

import numpy as np

from .params import PostprocessParams

# torchvision RetinaNet defaults (theory.md 2.3): anchor sizes per level, ratios, head limits.
_BASE_SIZES = (32, 64, 128, 256, 512)
_SCALES = (1.0, 2 ** (1 / 3), 2 ** (2 / 3))
_RATIOS = (0.5, 1.0, 2.0)
_BBOX_XFORM_CLIP = math.log(1000.0 / 16)
TOPK_CANDIDATES = 1000
DETECTIONS_PER_IMG = 300


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
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
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def _cell_anchors(size: int) -> np.ndarray:
    """torchvision `AnchorGenerator.generate_anchors` for one level -> (9, 4) centred at origin."""
    # torchvision truncates sizes to int: (32, 40, 50), (64, 80, 101), ...
    scales = np.asarray([size] + [int(size * s) for s in _SCALES[1:]], dtype=np.float32)
    ratios = np.asarray(_RATIOS, dtype=np.float32)
    h_ratios = np.sqrt(ratios)
    w_ratios = 1.0 / h_ratios
    ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)
    return np.round(np.stack([-ws, -hs, ws, hs], axis=1) / 2.0)


def level_grid_sizes(input_h: int, input_w: int) -> list[tuple[int, int]]:
    """Feature-map sizes of P3..P7 for a (32-divisible) padded input."""
    p5 = (input_h // 32, input_w // 32)
    p6 = (math.ceil(p5[0] / 2), math.ceil(p5[1] / 2))
    p7 = (math.ceil(p6[0] / 2), math.ceil(p6[1] / 2))
    return [(input_h // 8, input_w // 8), (input_h // 16, input_w // 16), p5, p6, p7]


def make_anchors(input_h: int, input_w: int) -> list[np.ndarray]:
    """Per-level anchors `(H*W*9, 4)` XYXY, same ordering as the RetinaNet head output.

    Strides use integer division of the padded input by the grid size, as torchvision does.
    """
    levels = []
    for size, (gh, gw) in zip(_BASE_SIZES, level_grid_sizes(input_h, input_w)):
        stride_h, stride_w = input_h // gh, input_w // gw
        shift_x = np.arange(gw, dtype=np.float32) * stride_w
        shift_y = np.arange(gh, dtype=np.float32) * stride_h
        sy, sx = np.meshgrid(shift_y, shift_x, indexing="ij")
        sx, sy = sx.reshape(-1), sy.reshape(-1)
        shifts = np.stack([sx, sy, sx, sy], axis=1)
        levels.append((shifts[:, None, :] + _cell_anchors(size)[None, :, :]).reshape(-1, 4))
    return levels


def _decode_boxes(rel_codes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    dx, dy = rel_codes[:, 0], rel_codes[:, 1]
    dw = np.minimum(rel_codes[:, 2], _BBOX_XFORM_CLIP)
    dh = np.minimum(rel_codes[:, 3], _BBOX_XFORM_CLIP)
    cx = dx * widths + ctr_x
    cy = dy * heights + ctr_y
    w = np.exp(dw) * widths
    h = np.exp(dh) * heights
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1)


def decode_retinanet_output(
    cls_logits: Any,
    bbox_regression: Any,
    input_hw: tuple[int, int],
    image_hw: tuple[int, int],
    cfg: PostprocessParams | None = None,
) -> list[dict[str, float | int]]:
    """Mirror of torchvision `RetinaNet.postprocess_detections` for one image.

    Args:
        cls_logits: `(1, A, C)` raw logits (all levels concatenated).
        bbox_regression: `(1, A, 4)` raw box deltas.
        input_hw: padded model input `(H, W)`; defines anchors and level split.
        image_hw: resized image size `(h, w)` inside the padding; boxes are clipped to it.
    Returns:
        Detections as dicts `x1,y1,x2,y2,confidence,class_id` in model-input pixels.
    """
    cfg = cfg or PostprocessParams()
    logits = np.asarray(cls_logits, dtype=np.float32)[0]
    deltas = np.asarray(bbox_regression, dtype=np.float32)[0]
    num_classes = logits.shape[-1]
    levels = make_anchors(*input_hw)
    if sum(len(a) for a in levels) != len(logits):
        raise ValueError(
            f"anchor count {sum(len(a) for a in levels)} != model outputs {len(logits)}; check input size"
        )

    boxes_all, scores_all, labels_all = [], [], []
    start = 0
    for anchors in levels:
        stop = start + len(anchors)
        scores = (1.0 / (1.0 + np.exp(-logits[start:stop]))).reshape(-1)
        idx = np.flatnonzero(scores > cfg.confidence_threshold)
        if idx.size > TOPK_CANDIDATES:
            idx = idx[np.argsort(-scores[idx], kind="stable")[:TOPK_CANDIDATES]]
        if idx.size:
            anchor_idx = idx // num_classes
            boxes = _decode_boxes(deltas[start:stop][anchor_idx], anchors[anchor_idx])
            h, w = image_hw
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
            boxes_all.append(boxes)
            scores_all.append(scores[idx])
            labels_all.append(idx % num_classes)
        start = stop

    if not boxes_all:
        return []
    boxes = np.concatenate(boxes_all)
    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)

    kept: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        kept.extend(int(members[i]) for i in nms_xyxy(boxes[members], scores[members], cfg.nms_threshold))
    kept.sort(key=lambda i: -scores[i])
    kept = kept[:DETECTIONS_PER_IMG]
    return [
        {
            "x1": float(boxes[i, 0]),
            "y1": float(boxes[i, 1]),
            "x2": float(boxes[i, 2]),
            "y2": float(boxes[i, 3]),
            "confidence": float(scores[i]),
            "class_id": int(labels[i]),
        }
        for i in kept
    ]


def merge_window_detections(
    detections: list[dict[str, float | int]], iou_threshold: float
) -> list[dict[str, float | int]]:
    """Cross-window NMS in chip-pixel space (DeepForest `predict.mosiac`), class-aware."""
    if len(detections) <= 1:
        return detections
    boxes = np.asarray([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=np.float32)
    scores = np.asarray([d["confidence"] for d in detections], dtype=np.float32)
    labels = np.asarray([d["class_id"] for d in detections])
    kept: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        kept.extend(int(members[i]) for i in nms_xyxy(boxes[members], scores[members], iou_threshold))
    kept.sort(key=lambda i: -scores[i])
    return [detections[i] for i in kept]
