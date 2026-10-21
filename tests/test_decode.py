import numpy as np

from td_hot.decode import decode_detections, nms_xyxy
from td_hot.params import PostprocessParams


def _cand(rows):
    """rows: (x1, y1, x2, y2, score, label)"""
    a = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return a[:, :4], a[:, 4], a[:, 5].astype(np.int64)


def test_empty_input_gives_no_detections():
    assert decode_detections(*_cand([])) == []


def test_confidence_filter_is_strict_and_tunable():
    c = _cand([[0, 0, 10, 10, 0.9, 0], [50, 50, 60, 60, 0.3, 0], [90, 90, 99, 99, 0.1, 0]])
    assert len(decode_detections(*c, PostprocessParams(confidence_threshold=0.1))) == 2  # 0.1 is dropped
    got = decode_detections(*c, PostprocessParams(confidence_threshold=0.5))
    assert len(got) == 1 and got[0]["confidence"] > 0.89


def test_nms_keeps_the_best_of_overlapping_boxes_and_threshold_is_tunable():
    c = _cand([[0, 0, 10, 10, 0.6, 0], [1, 1, 11, 11, 0.9, 0], [100, 100, 110, 110, 0.4, 0]])  # IoU ~0.68
    got = decode_detections(*c, PostprocessParams(confidence_threshold=0.1, nms_threshold=0.05))
    assert [round(d["confidence"], 1) for d in got] == [0.9, 0.4]
    got = decode_detections(*c, PostprocessParams(confidence_threshold=0.1, nms_threshold=0.9))
    assert len(got) == 3  # permissive NMS keeps the overlapping box too


def test_nms_is_per_class():
    c = _cand([[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 1]])
    assert len(decode_detections(*c)) == 2


def test_output_is_sorted_and_carries_fields():
    c = _cand([[0, 0, 10, 10, 0.4, 0], [50, 50, 60, 60, 0.9, 0]])
    got = decode_detections(*c)
    assert [round(d["confidence"], 1) for d in got] == [0.9, 0.4]
    assert set(got[0]) == {"x1", "y1", "x2", "y2", "confidence", "class_id"} and got[0]["class_id"] == 0


def test_cap_of_300_detections():
    xs = np.arange(400, dtype=np.float32) * 20
    rows = [[x, 0, x + 10, 10, 0.5 + x / 1e5, 0] for x in xs]  # 400 disjoint boxes
    assert len(decode_detections(*_cand(rows))) == 300


def test_nms_xyxy_keeps_highest_score_first():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.5, 0.9, 0.4], np.float32)
    assert nms_xyxy(boxes, scores, 0.3) == [1, 2]
    assert nms_xyxy(np.zeros((0, 4)), np.zeros(0), 0.3) == []
