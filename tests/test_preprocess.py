import numpy as np
import pytest

from td_hot.params import PreprocessParams
from td_hot.preprocess import preprocess_array, preprocess_chip, preprocess_window, sliding_windows


def test_single_window_when_chip_smaller_than_patch():
    assert sliding_windows(256, 256, 400, 0.05) == [(0, 0, 256, 256)]


def test_windows_match_slidingwindow_semantics():
    # 1000 px, patch 400, overlap 5% -> step 400 - 20 = 380; last window shifted back to the edge.
    wins = sliding_windows(1000, 400, 400, 0.05)
    assert sorted({x for x, _, _, _ in wins}) == [0, 380, 600]
    assert all(w == 400 and h == 400 for _, _, w, h in wins)
    assert max(x + w for x, _, w, _ in wins) == 1000  # full coverage


def test_zero_overlap_exact_multiple():
    assert sliding_windows(800, 800, 400, 0.0) == [
        (0, 0, 400, 400),
        (0, 400, 400, 400),
        (400, 0, 400, 400),
        (400, 400, 400, 400),
    ]


def test_window_tensor_shape_dtype_and_resize_to_800():
    rgb = np.full((400, 400, 3), 128, dtype=np.uint8)
    batch, new_w, new_h = preprocess_window(rgb)
    assert batch.shape == (1, 3, 800, 800) and batch.dtype == np.float32
    assert (new_w, new_h) == (800, 800)
    expected = (128 / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(batch[0, :, 400, 400], expected, atol=1e-5)


def test_non_square_window_is_padded_to_multiple_of_32():
    batch, new_w, new_h = preprocess_window(np.zeros((300, 420, 3), dtype=np.uint8))
    assert (new_h, new_w) == (800, 1120)
    assert batch.shape == (1, 3, 800, 1120)  # both already multiples of 32
    batch, new_w, new_h = preprocess_window(np.zeros((300, 410, 3), dtype=np.uint8))
    assert (new_h, new_w) == (800, 1093) and batch.shape[3] == 1120
    assert np.all(batch[0, :, :, new_w:] == 0)  # padding is zero


def test_long_side_is_capped_at_1333():
    _, new_w, new_h = preprocess_window(np.zeros((200, 1000, 3), dtype=np.uint8))
    assert new_w == 1333 and new_h == 266


def test_rejects_non_rgb_arrays():
    with pytest.raises(ValueError):
        preprocess_window(np.zeros((10, 10, 4), dtype=np.uint8))


def test_preprocess_array_meta_maps_back_to_source():
    rgb = np.zeros((500, 700, 3), dtype=np.uint8)
    batches, meta = preprocess_array(rgb, PreprocessParams(400, 0.05))
    assert len(batches) == len(meta.windows) and (meta.width, meta.height) == (700, 500)
    w = meta.windows[0]
    assert (w.scale_x, w.scale_y) == (2.0, 2.0)


def test_preprocess_chip_reads_geotiff(write_chip):
    path = write_chip(height=300, width=300)
    batches, meta = preprocess_chip(path)
    assert len(batches) == 1 and batches[0].shape == (1, 3, 800, 800)
    assert str(meta.crs) == "EPSG:3857" and meta.transform.a == 1.0


def test_preprocess_chip_rejects_fewer_than_three_bands(write_chip):
    path = write_chip(count=1)
    with pytest.raises(ValueError, match="3 bands"):
        preprocess_chip(path)
