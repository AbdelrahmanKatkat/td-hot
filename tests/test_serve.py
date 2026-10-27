import numpy as np
import pytest
from pyproj import Transformer

from conftest import CHIP_NAME, FakeSession
from td_hot.georef import box_to_chip_pixels, xyxy_to_epsg4326_polygon
from td_hot.preprocess import PreprocessMeta
from td_hot.serve import list_images, predict_session
from td_hot.tiles import chip_scale


def test_empty_predictions_give_valid_empty_collection(write_chip, tmp_path):
    write_chip(CHIP_NAME)
    fc = predict_session(FakeSession(), tmp_path, None)
    assert fc == {"type": "FeatureCollection", "features": []}


def test_chip_is_scaled_by_its_zoom_and_planted_box_lands_on_the_right_ground(write_chip, tmp_path):
    write_chip(CHIP_NAME)
    session = FakeSession(plant=True)
    fc = predict_session(session, tmp_path, {})

    scale = chip_scale(CHIP_NAME, 256, 0.05)
    side = round(256 * scale)
    assert session.calls == [(3, side, side)]  # one pass, one resize, CHW

    assert len(fc["features"]) == 1
    f = fc["features"][0]
    assert f["geometry"]["type"] == "Polygon" and len(f["geometry"]["coordinates"][0]) == 5
    assert f["properties"]["class_name"] == "Tree" and f["properties"]["source"] == CHIP_NAME
    assert 0.9 < f["properties"]["confidence"] <= 1.0

    # Planted box (100, 100)-(200, 200) in model pixels -> chip pixels -> EPSG:3857 -> lon/lat.
    # The chip fixture has its origin at (1_000_000, 2_000_000) with 1 m pixels.
    ratio = side / 256
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    expected = to_ll.transform(1_000_000 + 100 / ratio, 2_000_000 - 100 / ratio)
    np.testing.assert_allclose(f["geometry"]["coordinates"][0][0], expected, atol=1e-9)


def test_confidence_threshold_alias_filters_everything(write_chip, tmp_path):
    write_chip(CHIP_NAME)
    fc = predict_session(FakeSession(plant=True), tmp_path, {"score_thresh": 0.999})
    assert fc["features"] == []


def test_model_gsd_param_changes_the_input_size(write_chip, tmp_path):
    write_chip(CHIP_NAME)
    session = FakeSession()
    predict_session(session, tmp_path, {"model_gsd_m": 0.10})
    assert session.calls[0][1] == round(256 * chip_scale(CHIP_NAME, 256, 0.10))


def test_non_oam_chip_name_fails_loudly(write_chip, tmp_path):
    write_chip("chip.tif")
    with pytest.raises(ValueError, match="OAM"):
        predict_session(FakeSession(), tmp_path, None)


def test_no_images_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path)


def test_resized_to_chip_inverts_resize():
    meta = PreprocessMeta(
        width=256, height=256, resized_width=800, resized_height=800, transform=None, crs=None
    )
    det = {"x1": 100.0, "y1": 40.0, "x2": 200.0, "y2": 120.0}
    assert box_to_chip_pixels(det, meta) == (32.0, 12.8, 64.0, 38.4)


def test_polygon_in_epsg4326_passthrough():
    from rasterio.transform import from_origin

    ring = xyxy_to_epsg4326_polygon(0, 0, 10, 10, from_origin(10.0, 50.0, 0.001, 0.001), "EPSG:4326")
    assert ring[0] == ring[-1] and ring[0] == [10.0, 50.0]
    np.testing.assert_allclose(ring[2], [10.01, 49.99])


def test_predict_session_logs_one_summary_line_not_one_per_chip(write_chip, tmp_path, caplog):
    import logging

    for i in range(3):
        write_chip(f"OAM-52428{i}-524288-20.tif")
    with caplog.at_level(logging.INFO, logger="td_hot.serve"):
        predict_session(FakeSession(plant=True), tmp_path, None)
    lines = [r.getMessage() for r in caplog.records if r.name == "td_hot.serve"]
    assert lines == ["predict_session: 3 chips, 3 detections in " + lines[0].rsplit(" in ", 1)[1]]


def test_detect_chip_returns_chip_pixel_detections_and_meta(write_chip, tmp_path):
    from td_hot.serve import detect_chip

    path = write_chip(CHIP_NAME)
    dets, meta = detect_chip(FakeSession(plant=True), path, None)
    ratio = meta.resized_width / 256
    assert len(dets) == 1 and meta.width == 256
    assert (dets[0]["x1"], dets[0]["y1"], dets[0]["x2"], dets[0]["y2"]) == pytest.approx(
        (100 / ratio, 100 / ratio, 200 / ratio, 200 / ratio)
    )
