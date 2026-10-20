from td_hot.evaluate import match_detections


def test_match_detections_counts_tp_fp_fn():
    preds = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "confidence": 0.9, "class_id": 0},
        {"x1": 100, "y1": 100, "x2": 110, "y2": 110, "confidence": 0.8, "class_id": 0},
    ]
    gts = [
        {"x1": 1, "y1": 1, "x2": 11, "y2": 11, "class_id": 0},
    ]
    tp, fp, fn = match_detections(preds, gts, iou_threshold=0.5)
    assert (tp, fp, fn) == (1, 1, 0)
