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


class FakeSession:
    """Stands in for onnxruntime: returns zero logits (-> no detections) or a planted detection."""

    class _In:
        name = "images"

    def __init__(self, plant: bool = False):
        self.plant = plant
        self.calls: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feeds):
        from td_hot.decode import make_anchors

        batch = feeds["images"]
        self.calls.append(batch.shape)
        n = sum(len(a) for a in make_anchors(batch.shape[2], batch.shape[3]))
        cls = np.full((1, n, 1), -10.0, dtype=np.float32)
        box = np.zeros((1, n, 4), dtype=np.float32)
        if self.plant:
            cls[0, 0, 0] = 5.0  # anchor 0 (level P3, top-left): sigmoid(5) ~ 0.993
        return [cls, box]
