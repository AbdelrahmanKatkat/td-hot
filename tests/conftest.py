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


@pytest.fixture
def prepared_dataset(write_chip, tmp_path):
    """Two zoom-20 chips and one box label, split into `prepared/{train,val}/{images,labels.json}`."""
    import json
    import shutil

    from td_hot.dataset import build_chip_labels, spatial_split

    chips, labels, out = tmp_path / "chips", tmp_path / "labels", tmp_path / "prepared"
    chips.mkdir()
    labels.mkdir()
    for name in ("OAM-524288-524288-20.tif", "OAM-524292-524288-20.tif"):
        (chips / name).write_bytes(write_chip(name=name).read_bytes())
    box = [[[1_000_010, 1_999_980], [1_000_040, 1_999_980], [1_000_040, 1_999_950], [1_000_010, 1_999_950]]]
    feature = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": box}}
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [feature],
    }
    (labels / "labels.geojson").write_text(json.dumps(collection), encoding="utf-8")
    all_labels = build_chip_labels(chips, labels)
    train, val = spatial_split(sorted(all_labels), 0.5, 42, block_size=4)
    for split, names in (("train", train), ("val", val)):
        (out / split / "images").mkdir(parents=True)
        for name in names:
            shutil.copy2(chips / name, out / split / "images" / name)
        (out / split / "labels.json").write_text(
            json.dumps({n: all_labels[n] for n in names}), encoding="utf-8"
        )
    return out


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
