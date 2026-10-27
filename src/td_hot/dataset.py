"""Dataset helpers for td-hot (numpy/rasterio/pyproj/Pillow only, no torch; safe to import at serving time).

- `list_chip_paths`, `spatial_split`: find the chips and split them into train/val.
- `load_labels`: a `labels.json` path or mapping -> the chip name -> boxes mapping.
- `build_chip_labels`: crown polygons, boxes or tree points (GeoJSON) -> per-chip pixel boxes, nothing copied.
- `scale_chip_and_boxes`, `flip_chip_and_boxes`: the per-chip steps of training, applied to an image and its boxes.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .params import PreprocessParams
from .preprocess import preprocess_rgb
from .tiles import OAM_TILE_RE, chip_scale

CHIP_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
LABEL_SUFFIXES = (".geojson", ".json")
DEFAULT_CROWN_M = 5.0  # side of the square crown of a tree point without a usable `diameter_crown`
METRES_PER_DEGREE = 111_320  # spherical approximation, within 1% of WGS84 latitude degrees


def list_chip_paths(root: str | Path) -> list[Path]:
    """RGB chips in a folder. Training and evaluation read any of them (the scale comes from the OAM
    name), but turning GeoJSON labels into pixel boxes needs georeferenced GeoTIFFs."""
    folder = Path(root)
    return sorted(path for pattern in CHIP_PATTERNS for path in folder.glob(pattern) if path.is_file())


def spatial_split(
    chip_names: list[str], val_ratio: float, seed: int, *, block_size: int = 4
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coords; whole `(x // K, y // K)` blocks go to one side.

    Neighbouring chips share trees cut at chip borders and similar canopy, so a random
    chip split leaks. Every name must be `OAM-x-y-z.ext`; anything else raises `ValueError`.
    Returns sorted `(train, val)` name lists.
    """
    blocks_map: dict[tuple[int, int], list[str]] = {}
    for name in chip_names:
        match = OAM_TILE_RE.match(name)
        if match is None:
            raise ValueError(f"chip name {name!r} does not match OAM-<x>-<y>-<z>.<ext>")
        blocks_map.setdefault(
            (int(match.group(1)) // block_size, int(match.group(2)) // block_size), []
        ).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(blocks_map)
    rng.shuffle(blocks)

    n_val_target = max(1, int(len(chip_names) * val_ratio)) if chip_names else 0

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        (val if len(val) < n_val_target else train).extend(blocks_map[block])
    return sorted(train), sorted(val)


def _find_labels_geojson(labels: str | Path) -> Path:
    """The labels file: the path itself if it is a `.geojson` / `.json` file, else the one such file under a folder."""
    root = Path(labels)
    if root.is_file() and root.suffix.lower() in LABEL_SUFFIXES:
        return root
    matches = sorted(root.rglob("*.geojson")) or sorted(root.rglob("*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .geojson (or .json) in {root}, found {len(matches)}")
    return matches[0]


def _parse_geojson_crs(feature_collection: dict[str, Any]) -> str:
    """CRS name from the legacy GeoJSON `crs` member; RFC 7946 files default to EPSG:4326."""
    crs = feature_collection.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties")
        if isinstance(props, dict) and isinstance(props.get("name"), str) and props["name"]:
            return props["name"]
    return "EPSG:4326"


def _iter_xy_coords(node: Any) -> list[tuple[float, float]]:
    """All `(x, y)` pairs in a GeoJSON `coordinates` structure (any nesting, extra dims ignored)."""
    pairs: list[tuple[float, float]] = []

    def _walk(item: Any) -> None:
        if not isinstance(item, (list, tuple)):
            return
        if len(item) >= 2 and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in item
        ):
            pairs.append((float(item[0]), float(item[1])))
            return
        for child in item:
            _walk(child)

    _walk(node)
    return pairs


def load_labels(
    labels: str | Path | Mapping[str, list[dict[str, Any]]],
) -> Mapping[str, list[dict[str, Any]]]:
    """A `labels.json` path or the mapping itself -> chip filename -> list of pixel boxes."""
    if isinstance(labels, (str, Path)):
        labels = json.loads(Path(labels).read_text(encoding="utf-8"))
    if not isinstance(labels, Mapping):
        raise ValueError("labels must map chip filename -> list of boxes")
    return labels


def _point_crown_bounds(
    lon: float, lat: float, properties: dict[str, Any]
) -> tuple[float, float, float, float]:
    """The square crown around a tree point as `(minx, miny, maxx, maxy)` in degrees, as OSM maps trees as points.

    The side is the `diameter_crown` tag in metres (read from `tags` when OSM nests them), or `DEFAULT_CROWN_M` when it
    is missing or not a positive number, for example free text such as "3 m".
    """
    tags = properties.get("tags") or properties
    try:
        side_m = float(tags.get("diameter_crown") or DEFAULT_CROWN_M)
    except (TypeError, ValueError):
        side_m = DEFAULT_CROWN_M
    if not (math.isfinite(side_m) and side_m > 0):
        side_m = DEFAULT_CROWN_M
    half_lat = side_m / 2 / METRES_PER_DEGREE
    half_lon = half_lat / math.cos(math.radians(lat))
    return lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat


def read_label_boxes(labels_geojson: str | Path) -> tuple[np.ndarray, str]:
    """Crown bounding boxes `(N, 4)` as `(minx, miny, maxx, maxy)` in the file CRS, plus that CRS.

    A polygon or box gives its bounds. A tree point gives a square crown (`_point_crown_bounds`), which takes the
    point as longitude and latitude in degrees.
    """
    collection = json.loads(Path(labels_geojson).read_text(encoding="utf-8"))
    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        raise ValueError("labels file must be a GeoJSON FeatureCollection")
    features = collection.get("features", [])
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection 'features' must be a list")
    bounds: list[tuple[float, float, float, float]] = []
    for feature in features:
        geom = feature.get("geometry") if isinstance(feature, dict) else None
        if isinstance(geom, dict) and geom.get("type") == "Point":
            lon, lat = geom["coordinates"][:2]
            bounds.append(_point_crown_bounds(lon, lat, feature.get("properties") or {}))
            continue
        coords = _iter_xy_coords(geom.get("coordinates")) if isinstance(geom, dict) else []
        if coords:
            xs, ys = zip(*coords)
            bounds.append((min(xs), min(ys), max(xs), max(ys)))
    return np.asarray(bounds, dtype=np.float64).reshape(-1, 4), _parse_geojson_crs(collection)


def _project_bounds(bounds: np.ndarray, src_crs: Any, dst_crs: Any) -> np.ndarray:
    """Reproject `(N, 4)` bounds by projecting all four corners and re-bounding."""
    if len(bounds) == 0 or src_crs is None or dst_crs is None or str(src_crs) == str(dst_crs):
        return bounds
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = np.stack([bounds[:, 0], bounds[:, 0], bounds[:, 2], bounds[:, 2]], axis=1)
    ys = np.stack([bounds[:, 1], bounds[:, 3], bounds[:, 1], bounds[:, 3]], axis=1)
    out_x, out_y = transformer.transform(xs, ys)
    return np.stack([out_x.min(axis=1), out_y.min(axis=1), out_x.max(axis=1), out_y.max(axis=1)], axis=1)


def chip_boxes(
    chip_path: str | Path,
    *,
    label_bounds: np.ndarray,
    labels_crs: Any,
    class_id: int = 0,
    min_box_px: float = 0.0,
) -> list[dict[str, float | int]]:
    """Label bounds -> XYXY boxes in chip pixels, clipped to the chip.

    Boxes off the chip are dropped, and so are boxes thinner than `min_box_px` after clipping.
    """
    import rasterio

    with rasterio.open(chip_path) as src:
        width, height, transform, crs = int(src.width), int(src.height), src.transform, src.crs
    if crs is None:
        raise ValueError(f"{chip_path} has no CRS; GeoJSON labels need a georeferenced GeoTIFF chip")
    if len(label_bounds) == 0:
        return []

    projected = _project_bounds(label_bounds, labels_crs, crs)
    corner_x = np.stack([projected[:, 0], projected[:, 0], projected[:, 2], projected[:, 2]], axis=1)
    corner_y = np.stack([projected[:, 1], projected[:, 3], projected[:, 1], projected[:, 3]], axis=1)
    inv = ~transform
    # Explicit affine maths (same reason as georef.py): works across `affine` versions.
    cols = inv.a * corner_x + inv.b * corner_y + inv.c
    rows = inv.d * corner_x + inv.e * corner_y + inv.f
    x1, x2 = np.clip(cols.min(axis=1), 0, width), np.clip(cols.max(axis=1), 0, width)
    y1, y2 = np.clip(rows.min(axis=1), 0, height), np.clip(rows.max(axis=1), 0, height)
    keep = (x2 > x1) & (y2 > y1) & (x2 - x1 >= min_box_px) & (y2 - y1 >= min_box_px)
    return [
        {
            "x1": float(left),
            "y1": float(top),
            "x2": float(right),
            "y2": float(bottom),
            "class_id": int(class_id),
        }
        for left, top, right, bottom in zip(x1[keep], y1[keep], x2[keep], y2[keep])
    ]


def build_chip_labels(
    chips_dir: str | Path, labels: str | Path, *, class_id: int = 0, min_box_px: float = 0.0
) -> dict[str, list[dict[str, float | int]]]:
    """Pixel boxes of every chip in `chips_dir` from one GeoJSON of polygon, box or tree-point labels.

    `labels` is the `.geojson` / `.json` file, or a folder holding exactly one. Nothing is copied, so a pipeline step
    can call this on the chips and labels it was given, with no prepared folder shared between isolated steps.
    Chips without a label get an empty list (negative examples); boxes thinner than `min_box_px` are dropped.
    """
    bounds, crs = read_label_boxes(_find_labels_geojson(labels))
    return {
        path.name: chip_boxes(
            path, label_bounds=bounds, labels_crs=crs, class_id=class_id, min_box_px=min_box_px
        )
        for path in list_chip_paths(chips_dir)
    }


def scale_chip_and_boxes(
    rgb: np.ndarray, boxes: np.ndarray, pre: PreprocessParams, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Scale an RGB chip exactly as at serving time, and its `(N, 4)` XYXY chip-pixel boxes with it.

    Returns the `(3, H', W')` float32 image and the scaled boxes as float32.
    """
    scale = chip_scale(
        name, rgb.shape[1], pre.model_gsd_m, height_px=rgb.shape[0], max_input_px=pre.max_input_px
    )
    image, meta = preprocess_rgb(rgb, scale)
    factors = np.asarray([meta.scale_x, meta.scale_y, meta.scale_x, meta.scale_y], dtype=np.float32)
    return image, np.asarray(boxes, dtype=np.float32) * factors


def flip_chip_and_boxes(image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a `(3, H, W)` image left-right, and its `(N, 4)` XYXY boxes with it."""
    flipped = np.ascontiguousarray(image[:, :, ::-1])
    flipped_boxes = np.asarray(boxes, dtype=np.float32).copy()
    flipped_boxes[:, [0, 2]] = image.shape[2] - boxes[:, [2, 0]]
    return flipped, flipped_boxes
