"""One tiny fine-tune + eval pass on random weights (CPU, a few seconds): needs the `train` extra."""

import json

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.integration

from td_hot.dataset import prepare_detection_dataset_from_geojson  # noqa: E402
from td_hot.evaluate import evaluate_retinanet  # noqa: E402
from td_hot.params import SplitParams, parse_train_params  # noqa: E402
from td_hot.train import train_retinanet  # noqa: E402


def test_train_and_evaluate_smoke(write_chip, tmp_path):
    chips, labels, out = tmp_path / "chips", tmp_path / "labels", tmp_path / "prepared"
    chips.mkdir()
    labels.mkdir()
    for name in ("OAM-0-0-19.tif", "OAM-4-0-19.tif"):
        (chips / name).write_bytes(write_chip(name=name).read_bytes())
    box = [[[1_000_010, 1_999_980], [1_000_040, 1_999_980], [1_000_040, 1_999_950], [1_000_010, 1_999_950]]]
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": box}}
        ],
    }
    (labels / "labels.geojson").write_text(json.dumps(fc), encoding="utf-8")
    prepare_detection_dataset_from_geojson(
        chips_dir=chips, labels_dir=labels, out_dir=out, split_cfg=SplitParams(val_ratio=0.5, block_size=4)
    )

    result = train_retinanet(
        train_images_dir=out / "train" / "images",
        train_labels_json=out / "train" / "labels.json",
        val_images_dir=out / "val" / "images",
        val_labels_json=out / "val" / "labels.json",
        cfg=parse_train_params({"epochs": 1, "device": "cpu", "freeze_backbone": True}),
    )
    assert len(result.history["train_loss"]) == 1 and result.best_loss > 0
    assert not result.model.training

    metrics = evaluate_retinanet(
        result.model, images_dir=out / "val" / "images", labels_json=out / "val" / "labels.json"
    )
    assert {"precision", "recall", "f1", "tp", "fp", "fn"} <= metrics.keys()
