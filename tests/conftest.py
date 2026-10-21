import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def write_chip(tmp_path):
    """Write a synthetic georeferenced RGB GeoTIFF (EPSG:3857, 1 m pixels)."""

    def _write(name: str = "chip.tif", height: int = 256, width: int = 256, count: int = 3):
        path = tmp_path / name
        data = np.random.default_rng(0).integers(0, 256, size=(count, height, width), dtype=np.uint8)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_origin(1_000_000.0, 2_000_000.0, 1.0, 1.0),
        ) as dst:
            dst.write(data)
        return path

    return _write


# A tile near the equator at zoom 20: 14.9 cm/px, so a 256 px chip is scaled by about 2.99.
CHIP_NAME = "OAM-524288-524288-20.tif"


class FakeSession:
    """Stands in for onnxruntime: returns no candidates, or one planted candidate (level-2 contract)."""

    class _In:
        name = "image"

    def __init__(self, plant: bool = False):
        self.plant = plant
        self.calls: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feeds):
        image = feeds["image"]
        assert image.ndim == 3 and image.dtype == np.float32 and 0.0 <= image.min() <= image.max() <= 1.0
        self.calls.append(image.shape)
        if not self.plant:
            return [np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)]
        return [
            np.asarray([[100.0, 100.0, 200.0, 200.0]], np.float32),  # model-input pixels
            np.asarray([0.99], np.float32),
            np.asarray([0], np.int64),
        ]
