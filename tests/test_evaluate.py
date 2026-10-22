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


def test_score_session_scores_the_deployed_path_without_torch(write_chip, tmp_path):
    from conftest import CHIP_NAME, FakeSession
    from td_hot.evaluate import score_session
    from td_hot.tiles import chip_scale

    write_chip(CHIP_NAME)
    ratio = round(256 * chip_scale(CHIP_NAME, 256, 0.05)) / 256
    truth = {"x1": 100 / ratio, "y1": 100 / ratio, "x2": 200 / ratio, "y2": 200 / ratio}
    hit = score_session(
        FakeSession(plant=True), images_dir=tmp_path, labels={CHIP_NAME: [truth]}, iou_thresholds=(0.5, 0.9)
    )
    assert hit[0.5]["tp"] == 1 and hit[0.5]["f1"] == 1.0 and hit[0.9]["tp"] == 1
    miss = score_session(FakeSession(plant=True), images_dir=tmp_path, labels={CHIP_NAME: []})
    assert miss[0.5]["fp"] == 1 and miss[0.5]["precision"] == 0.0
    none = score_session(FakeSession(), images_dir=tmp_path, labels={CHIP_NAME: [truth]})
    assert none[0.5]["fn"] == 1 and none[0.5]["recall"] == 0.0
    assert (
        score_session(
            FakeSession(plant=True), images_dir=tmp_path, labels={CHIP_NAME: [truth]}, names=["other.tif"]
        )[0.5]["tp"]
        == 0
    )
