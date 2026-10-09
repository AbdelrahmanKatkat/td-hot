import numpy as np
import pytest

from td_hot.decode import (
    decode_retinanet_output,
    level_grid_sizes,
    make_anchors,
    merge_window_detections,
    nms_xyxy,
)
from td_hot.params import PostprocessParams


def test_anchor_counts_for_800():
    assert level_grid_sizes(800, 800) == [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]
    assert [len(a) for a in make_anchors(800, 800)] == [90000, 22500, 5625, 1521, 441]  # x 9 anchors


def test_first_cell_anchors_use_truncated_sizes():
    a = make_anchors(800, 800)[0][:9]
    widths = (a[:, 2] - a[:, 0]).tolist()
    # ratios (0.5, 1, 2) x sizes (32, 40, 50) with torchvision rounding of the half-extents
    assert widths[3:6] == [32.0, 40.0, 50.0]


def _outputs(n_anchors: int):
    return np.full((1, n_anchors, 1), -10.0, np.float32), np.zeros((1, n_anchors, 4), np.float32)


def test_empty_when_nothing_above_threshold():
    n = sum(len(a) for a in make_anchors(800, 800))
    cls, box = _outputs(n)
    assert decode_retinanet_output(cls, box, (800, 800), (800, 800)) == []


def test_zero_delta_returns_the_anchor_box_clipped():
    anchors = make_anchors(800, 800)
    n = sum(len(a) for a in anchors)
    cls, box = _outputs(n)
    cls[0, 0, 0] = 4.0  # first P3 anchor, ratio 0.5 size 32 -> box centred on (0, 0)
    dets = decode_retinanet_output(cls, box, (800, 800), (800, 800))
    assert len(dets) == 1
    d = dets[0]
    assert (d["x1"], d["y1"]) == (0.0, 0.0)  # clipped at the image border
    assert d["x2"] == pytest.approx(anchors[0][0, 2]) and d["y2"] == pytest.approx(anchors[0][0, 3])
    assert d["confidence"] == pytest.approx(1 / (1 + np.exp(-4.0)), rel=1e-5) and d["class_id"] == 0


def test_box_deltas_shift_and_scale():
    anchors = make_anchors(800, 800)
    n = sum(len(a) for a in anchors)
    cls, box = _outputs(n)
    idx = 9 * (50 * 100 + 50) + 4  # P3 cell (y=50, x=50), ratio 1.0 / size 32 -> square 32x32
    a = anchors[0][idx]
    cls[0, idx, 0] = 4.0
    box[0, idx] = [0.5, 0.0, np.log(2.0), 0.0]  # shift by half a width, double the width
    d = decode_retinanet_output(cls, box, (800, 800), (800, 800))[0]
    w = a[2] - a[0]
    cx = (a[0] + a[2]) / 2 + 0.5 * w
    assert (d["x2"] - d["x1"]) == pytest.approx(2 * w, rel=1e-5)
    assert (d["x1"] + d["x2"]) / 2 == pytest.approx(cx, rel=1e-5)


def test_threshold_and_per_window_nms():
    anchors = make_anchors(800, 800)
    n = sum(len(a) for a in anchors)
    cls, box = _outputs(n)
    cls[0, 9 * (50 * 100 + 50) + 4, 0] = 4.0
    cls[0, 9 * (50 * 100 + 51) + 4, 0] = 3.0  # neighbouring cell, strongly overlapping box
    cfg = PostprocessParams(confidence_threshold=0.1, nms_threshold=0.9, iou_threshold=0.15)
    assert len(decode_retinanet_output(cls, box, (800, 800), (800, 800), cfg)) == 2
    cfg = PostprocessParams(confidence_threshold=0.1, nms_threshold=0.05, iou_threshold=0.15)
    assert (
        len(decode_retinanet_output(cls, box, (800, 800), (800, 800), cfg)) == 1
    )  # 8 px apart: IoU 0.6 > 0.05
    cfg = PostprocessParams(confidence_threshold=0.99, nms_threshold=0.05, iou_threshold=0.15)
    assert decode_retinanet_output(cls, box, (800, 800), (800, 800), cfg) == []


def test_output_size_mismatch_raises():
    cls, box = _outputs(100)
    with pytest.raises(ValueError, match="anchor count"):
        decode_retinanet_output(cls, box, (800, 800), (800, 800))


def test_nms_xyxy_keeps_highest_score_first():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.5, 0.9, 0.4], np.float32)
    assert nms_xyxy(boxes, scores, 0.3) == [1, 2]
    assert nms_xyxy(np.zeros((0, 4)), np.zeros(0), 0.3) == []


def test_merge_window_detections():
    dets = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "confidence": 0.6, "class_id": 0},
        {"x1": 1, "y1": 1, "x2": 11, "y2": 11, "confidence": 0.9, "class_id": 0},
        {"x1": 100, "y1": 100, "x2": 110, "y2": 110, "confidence": 0.4, "class_id": 0},
    ]
    merged = merge_window_detections(dets, 0.15)
    assert [d["confidence"] for d in merged] == [0.9, 0.4]
    assert merge_window_detections([], 0.15) == [] and merge_window_detections(dets[:1], 0.15) == dets[:1]
