import numpy as np
import pytest

from td_hot.dataset import flip_chip_and_boxes, scale_chip_and_boxes, spatial_split
from td_hot.params import PreprocessParams


def _names(n_x: int, n_y: int):
    return [f"OAM-{x}-{y}-19.tif" for x in range(n_x) for y in range(n_y)]


def test_split_is_deterministic_and_disjoint():
    names = _names(16, 16)
    a = spatial_split(names, 0.2, 1)
    assert a == spatial_split(names, 0.2, 1) and a != spatial_split(names, 0.2, 2)
    train, val = a
    assert set(train).isdisjoint(val) and sorted(train + val) == sorted(names)


def test_whole_blocks_stay_together():
    train, val = spatial_split(_names(16, 16), 0.25, 0, block_size=4)
    block = lambda n: tuple(int(v) // 4 for v in n.split("-")[1:3])  # noqa: E731
    assert {block(n) for n in train}.isdisjoint({block(n) for n in val})


def test_non_oam_name_raises_and_empty_input():
    with pytest.raises(ValueError, match="OAM"):
        spatial_split(["OAM-1-2-3.tif", "a.tif"], 0.4, 0)
    assert spatial_split([], 0.2, 0) == ([], [])


def test_flip_mirrors_image_and_boxes_and_keeps_the_input():
    image = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    boxes = np.array([[1, 0, 3, 2]], dtype=np.float32)
    flipped, flipped_boxes = flip_chip_and_boxes(image, boxes)
    assert np.array_equal(flipped, image[:, :, ::-1])
    assert flipped_boxes.tolist() == [[3, 0, 5, 2]]  # 6 - 3, 6 - 1
    assert boxes.tolist() == [[1, 0, 3, 2]]


def test_scale_follows_the_zoom_scale_for_image_and_boxes():
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
    image, scaled = scale_chip_and_boxes(rgb, boxes, PreprocessParams(), "OAM-485915-499567-20.tif")
    assert image.shape[0] == 3 and image.dtype == np.float32 and image.shape[1] > 256
    ratio = image.shape[1] / 256
    assert scaled.dtype == np.float32 and np.allclose(scaled, boxes * ratio)
