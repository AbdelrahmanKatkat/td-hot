import json

import numpy as np
import pytest

from td_hot.dataset import DEFAULT_CROWN_M, METRES_PER_DEGREE, _point_crown_bounds, build_chip_labels


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


def _labels(tmp_path, write_chip, labels, names=("OAM-0-0-19.tif", "OAM-4-0-19.tif"), **options):
    chips_dir, labels_dir = tmp_path / "chips", tmp_path / "labels"
    chips_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for name in names:
        (chips_dir / name).write_bytes(write_chip(name=name).read_bytes())
    (labels_dir / "labels.geojson").write_text(json.dumps(labels), encoding="utf-8")
    return build_chip_labels(chips_dir, labels_dir, **options)


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

    from td_hot.dataset import chip_boxes

    png = tmp_path / "OAM-524288-524288-20.png"
    Image.new("RGB", (256, 256)).save(png)
    with pytest.raises(ValueError, match="no CRS"):
        chip_boxes(png, label_bounds=np.zeros((1, 4)), labels_crs="EPSG:3857")


def test_labels_file_may_be_json_and_nested_in_a_folder(tmp_path):
    from td_hot.dataset import _find_labels_geojson

    nested = tmp_path / "labels" / "download"
    nested.mkdir(parents=True)
    (nested / "trees.json").write_text("{}", encoding="utf-8")
    assert _find_labels_geojson(tmp_path / "labels") == nested / "trees.json"
    assert _find_labels_geojson(nested / "trees.json") == nested / "trees.json"


def test_boxes_thinner_than_min_box_px_are_dropped(write_chip, tmp_path):
    labels = _labels_geojson_epsg3857([(1_000_010.0, 1_999_980.0, 1_000_011.5, 1_999_990.0)])  # 1.5 px wide
    kept = _labels(tmp_path / "default", write_chip, labels)
    dropped = _labels(tmp_path / "strict", write_chip, labels, min_box_px=2.0)
    assert all(len(rows) == 1 for rows in kept.values())
    assert all(rows == [] for rows in dropped.values())


def _side_degrees(properties):
    min_x, min_y, max_x, max_y = _point_crown_bounds(0.0, 0.0, properties)
    return max_x - min_x, max_y - min_y


def test_a_tree_point_is_a_square_crown_of_its_diameter():
    assert _side_degrees({"diameter_crown": "10"}) == pytest.approx((10 / METRES_PER_DEGREE,) * 2)
    assert _side_degrees({"tags": {"natural": "tree", "diameter_crown": "10"}}) == pytest.approx(
        (10 / METRES_PER_DEGREE,) * 2
    )


@pytest.mark.parametrize(
    "diameter", [None, "3 m", "nan", "-3", "0"], ids=["missing", "free text", "nan", "negative", "zero"]
)
def test_a_tree_point_without_a_usable_diameter_takes_the_default_crown(diameter):
    tags = {"natural": "tree"} if diameter is None else {"natural": "tree", "diameter_crown": diameter}
    assert _side_degrees({"tags": tags}) == pytest.approx((DEFAULT_CROWN_M / METRES_PER_DEGREE,) * 2)


def test_a_crown_away_from_the_equator_is_square_on_the_ground():
    min_x, min_y, max_x, max_y = _point_crown_bounds(0.0, 60.0, {"diameter_crown": "10"})
    assert (max_x - min_x) == pytest.approx(
        2 * (max_y - min_y)
    )  # a degree of longitude is half as long at 60 degrees


def test_tree_points_give_boxes_on_their_chip(write_chip, tmp_path):
    from pyproj import Transformer

    lon, lat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
        1_000_015.0, 1_999_985.0
    )
    point = {
        "type": "Feature",
        "properties": {"tags": {"natural": "tree", "diameter_crown": "10"}},
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }
    boxes = _labels(tmp_path, write_chip, {"type": "FeatureCollection", "features": [point]})
    for rows in boxes.values():
        (box,) = rows
        assert (box["x1"] + box["x2"]) / 2 == pytest.approx(15, abs=0.5)  # centred on the point
        assert (
            10 < box["x2"] - box["x1"] < 11
        )  # 10 ground metres, about 10.5 Web Mercator pixels at this latitude
