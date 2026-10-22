import json

import numpy as np
import pytest

from td_hot.dataset import labels_for_chips


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


def test_load_label_bounds_rejects_multiple_geojson(tmp_path):
    from td_hot.dataset import _find_labels_geojson

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "a.geojson").write_text("{}", encoding="utf-8")
    (labels_dir / "b.geojson").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected exactly one"):
        _find_labels_geojson(labels_dir)


def _labels(tmp_path, write_chip, labels, names=("OAM-0-0-19.tif", "OAM-4-0-19.tif")):
    chips_dir, labels_dir = tmp_path / "chips", tmp_path / "labels"
    chips_dir.mkdir()
    labels_dir.mkdir()
    for name in names:
        (chips_dir / name).write_bytes(write_chip(name=name).read_bytes())
    (labels_dir / "labels.geojson").write_text(json.dumps(labels), encoding="utf-8")
    return labels_for_chips(chips_dir, labels_dir)


def test_box_pixels_match_chip_geometry(write_chip, tmp_path):
    # Chip origin (1_000_000, 2_000_000), 1 m pixels: x 1_000_010..20 -> cols 10..20, y 1_999_980..90 -> rows 10..20.
    labels = _labels_geojson_epsg3857([(1_000_010.0, 1_999_980.0, 1_000_020.0, 1_999_990.0)])
    boxes = _labels(tmp_path, write_chip, labels)
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
    for chip_boxes in _labels(tmp_path, write_chip, labels).values():
        assert chip_boxes == [{"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "class_id": 0}]


def test_labels_in_epsg4326_are_reprojected(write_chip, tmp_path):
    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    (lon1, lat1), (lon2, lat2) = (
        to_ll.transform(1_000_010.0, 1_999_980.0),
        to_ll.transform(1_000_020.0, 1_999_990.0),
    )
    labels = _labels_geojson_epsg3857([(lon1, lat1, lon2, lat2)], epsg="EPSG:4326")
    box = next(iter(_labels(tmp_path, write_chip, labels).values()))[0]
    assert [round(box[k]) for k in ("x1", "y1", "x2", "y2")] == [10, 10, 20, 20]


def test_chip_without_a_label_gets_an_empty_list(write_chip, tmp_path):
    labels = _labels_geojson_epsg3857([(5_000_000.0, 5_000_000.0, 5_000_010.0, 5_000_010.0)])
    assert _labels(tmp_path, write_chip, labels) == {"OAM-0-0-19.tif": [], "OAM-4-0-19.tif": []}


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_chip_without_georeferencing_is_rejected(tmp_path):
    from PIL import Image

    from td_hot.dataset import boxes_for_chip

    png = tmp_path / "OAM-524288-524288-20.png"
    Image.new("RGB", (256, 256)).save(png)
    with pytest.raises(ValueError, match="no CRS"):
        boxes_for_chip(png, label_bounds=np.zeros((1, 4)), labels_crs="EPSG:3857")
