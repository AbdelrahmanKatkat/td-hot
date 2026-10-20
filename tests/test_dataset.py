import pytest

from td_hot.dataset import spatial_split


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
