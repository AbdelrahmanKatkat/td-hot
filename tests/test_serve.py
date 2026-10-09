import numpy as np
import pytest

from conftest import FakeSession
from td_hot.georef import window_to_chip, xyxy_to_epsg4326_polygon
from td_hot.preprocess import WindowMeta
from td_hot.serve import iter_image_paths, predict_session


def test_empty_predictions_give_valid_empty_collection(write_chip, tmp_path):
    write_chip("a.tif")
    fc = predict_session(FakeSession(), tmp_path, None)
    assert fc == {"type": "FeatureCollection", "features": []}


def test_planted_detection_becomes_georeferenced_polygon(write_chip, tmp_path):
    write_chip("OAM-1-2-3.tif", height=256, width=256)
    session = FakeSession(plant=True)
    fc = predict_session(session, tmp_path, {})
    assert session.calls == [(1, 3, 800, 800)]  # 256 px chip -> one window, resized to 800
    assert len(fc["features"]) == 1
    f = fc["features"][0]
    assert f["geometry"]["type"] == "Polygon" and len(f["geometry"]["coordinates"][0]) == 5
    assert f["properties"]["class_name"] == "Tree" and f["properties"]["source"] == "OAM-1-2-3.tif"
    lon, lat = f["geometry"]["coordinates"][0][0]
    assert -180 <= lon <= 180 and -90 <= lat <= 90  # converted from EPSG:3857 to 4326
    assert 0.9 < f["properties"]["confidence"] <= 1.0


def test_large_chip_runs_multiple_windows_and_merges(write_chip, tmp_path):
    write_chip("big.tif", height=500, width=900)
    session = FakeSession(plant=True)
    fc = predict_session(session, tmp_path, {"patch_size": 400, "patch_overlap": 0.05})
    assert len(session.calls) == 3 * 2  # x offsets 0, 380, 500 ; y offsets 0, 100
    # Planted box sits at each window's top-left; distinct windows -> distinct (non-overlapping) boxes survive NMS.
    assert 1 <= len(fc["features"]) <= len(session.calls)


def test_confidence_threshold_alias_filters_everything(write_chip, tmp_path):
    write_chip("a.tif")
    fc = predict_session(FakeSession(plant=True), tmp_path, {"score_thresh": 0.999})
    assert fc["features"] == []


def test_no_images_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        iter_image_paths(tmp_path)


def test_window_to_chip_inverts_resize_and_offset():
    w = WindowMeta(x=380, y=100, width=400, height=400, resized_width=800, resized_height=800)
    det = {"x1": 80.0, "y1": 40.0, "x2": 160.0, "y2": 120.0}
    assert window_to_chip(det, w) == (420.0, 120.0, 460.0, 160.0)


def test_polygon_in_epsg4326_passthrough():
    from rasterio.transform import from_origin

    ring = xyxy_to_epsg4326_polygon(0, 0, 10, 10, from_origin(10.0, 50.0, 0.001, 0.001), "EPSG:4326")
    assert ring[0] == ring[-1] and ring[0] == [10.0, 50.0]
    np.testing.assert_allclose(ring[2], [10.01, 49.99])
