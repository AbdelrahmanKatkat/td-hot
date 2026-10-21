import json

import numpy as np
import pytest

from td_hot.dataset import prepare_detection_dataset_from_geojson
from td_hot.params import SplitParams


def _labels_geojson_epsg3857(boxes_crs, epsg: str = "EPSG:3857") -> dict:
    """One polygon feature per (minx,miny,maxx,maxy) bbox in the given CRS."""
    feats = []
    for minx, miny, maxx, maxy in boxes_crs:
        feats.append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]],
                    ],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": epsg}},
        "features": feats,
    }


def test_prepare_detection_dataset_from_geojson_builds_boxes(write_chip, tmp_path):
    chips_dir = tmp_path / "chips"
    labels_dir = tmp_path / "labels"
    out_dir = tmp_path / "prepared"
    chips_dir.mkdir()
    labels_dir.mkdir()

    chip_a = write_chip(name="OAM-0-0-19.tif")
    chip_b = write_chip(name="OAM-4-0-19.tif")
    (chips_dir / chip_a.name).write_bytes(chip_a.read_bytes())
    (chips_dir / chip_b.name).write_bytes(chip_b.read_bytes())

    # Chip transform in conftest: origin (1_000_000, 2_000_000), 1m pixels. Pixel (10,10) is x=1_000_010, y=1_999_990.
    labels = _labels_geojson_epsg3857([(1_000_010.0, 1_999_980.0, 1_000_020.0, 1_999_990.0)])
    (labels_dir / "labels.geojson").write_text(json.dumps(labels), encoding="utf-8")

    info = prepare_detection_dataset_from_geojson(
        chips_dir=chips_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        split_cfg=SplitParams(val_ratio=0.5, split_seed=0, block_size=4),
    )
    assert info["strategy"] == "spatial"
    assert info["train_count"] + info["val_count"] == 2
    assert (out_dir / "train" / "images").is_dir()
    assert (out_dir / "val" / "images").is_dir()

    # Both splits write a labels.json mapping filename -> list of boxes.
    for split in ("train", "val"):
        labels_path = out_dir / split / "labels.json"
        assert labels_path.is_file()
        mapping = json.loads(labels_path.read_text(encoding="utf-8"))
        assert isinstance(mapping, dict)
        for name, boxes in mapping.items():
            assert isinstance(name, str)
            assert isinstance(boxes, list)

    # At least one split should contain the labeled chip and therefore have a non-empty box list.
    any_boxes = 0
    for split in ("train", "val"):
        mapping = json.loads((out_dir / split / "labels.json").read_text(encoding="utf-8"))
        any_boxes += sum(len(v) for v in mapping.values())
    assert any_boxes >= 1


def test_load_label_bounds_rejects_multiple_geojson(tmp_path):
    from td_hot.dataset import _find_labels_geojson

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "a.geojson").write_text("{}", encoding="utf-8")
    (labels_dir / "b.geojson").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected exactly one"):
        _find_labels_geojson(labels_dir)


def _prepare(tmp_path, write_chip, labels, names=("OAM-0-0-19.tif", "OAM-4-0-19.tif")):
    chips_dir, labels_dir = tmp_path / "chips", tmp_path / "labels"
    chips_dir.mkdir()
    labels_dir.mkdir()
    for name in names:
        (chips_dir / name).write_bytes(write_chip(name=name).read_bytes())
    (labels_dir / "labels.geojson").write_text(json.dumps(labels), encoding="utf-8")
    out_dir = tmp_path / "prepared"
    info = prepare_detection_dataset_from_geojson(
        chips_dir=chips_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        split_cfg=SplitParams(val_ratio=0.5, split_seed=0, block_size=4),
    )
    return info, out_dir


def _all_boxes(out_dir):
    boxes = {}
    for split in ("train", "val"):
        boxes.update(json.loads((out_dir / split / "labels.json").read_text(encoding="utf-8")))
    return boxes


def test_box_pixels_match_chip_geometry(write_chip, tmp_path):
    # Chip origin (1_000_000, 2_000_000), 1 m pixels: x 1_000_010..20 -> cols 10..20, y 1_999_980..90 -> rows 10..20.
    labels = _labels_geojson_epsg3857([(1_000_010.0, 1_999_980.0, 1_000_020.0, 1_999_990.0)])
    _, out_dir = _prepare(tmp_path, write_chip, labels)
    boxes = _all_boxes(out_dir)
    assert len(boxes) == 2
    # The synthetic chips share one footprint, so both contain the crown.
    for chip_boxes in boxes.values():
        assert chip_boxes == [{"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 20.0, "class_id": 0}]


def test_boxes_are_clipped_and_off_chip_labels_dropped(write_chip, tmp_path):
    labels = _labels_geojson_epsg3857(
        [
            (999_990.0, 1_999_990.0, 1_000_010.0, 2_000_010.0),  # straddles the top-left corner
            (5_000_000.0, 5_000_000.0, 5_000_010.0, 5_000_010.0),  # far away
        ]
    )
    _, out_dir = _prepare(tmp_path, write_chip, labels)
    for chip_boxes in _all_boxes(out_dir).values():
        assert chip_boxes == [{"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "class_id": 0}]


def test_labels_in_epsg4326_are_reprojected(write_chip, tmp_path):
    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    (lon1, lat1), (lon2, lat2) = (
        to_ll.transform(1_000_010.0, 1_999_980.0),
        to_ll.transform(1_000_020.0, 1_999_990.0),
    )
    labels = _labels_geojson_epsg3857([(lon1, lat1, lon2, lat2)], epsg="EPSG:4326")
    _, out_dir = _prepare(tmp_path, write_chip, labels)
    box = next(iter(_all_boxes(out_dir).values()))[0]
    assert [round(box[k]) for k in ("x1", "y1", "x2", "y2")] == [10, 10, 20, 20]


def test_non_empty_output_dir_is_rejected(write_chip, tmp_path):
    labels = _labels_geojson_epsg3857([(1_000_010.0, 1_999_980.0, 1_000_020.0, 1_999_990.0)])
    _, out_dir = _prepare(tmp_path, write_chip, labels)
    with pytest.raises(FileExistsError):
        prepare_detection_dataset_from_geojson(
            chips_dir=tmp_path / "chips",
            labels_dir=tmp_path / "labels",
            out_dir=out_dir,
            split_cfg=SplitParams(val_ratio=0.5, split_seed=1, block_size=4),
        )


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_chip_without_georeferencing_is_rejected(tmp_path):
    from PIL import Image

    from td_hot.dataset import boxes_for_chip

    png = tmp_path / "OAM-524288-524288-20.png"
    Image.new("RGB", (256, 256)).save(png)
    with pytest.raises(ValueError, match="no CRS"):
        boxes_for_chip(png, label_bounds=np.zeros((1, 4)), labels_crs="EPSG:3857")
