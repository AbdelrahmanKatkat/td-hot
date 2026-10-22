"""Dataset helpers for td-hot (numpy/rasterio/pyproj only; safe to import at serving time).

- `spatial_split`: deterministic block split of OAM chips into train/val.
- `labels_for_chips`: crown polygons or boxes (GeoJSON) -> per-chip pixel boxes, nothing copied.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .tiles import OAM_TILE_RE


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
        m = OAM_TILE_RE.match(name)
        if m is None:
            raise ValueError(f"chip name {name!r} does not match OAM-<x>-<y>-<z>.<ext>")
        blocks_map.setdefault((int(m.group(1)) // block_size, int(m.group(2)) // block_size), []).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(blocks_map)
    rng.shuffle(blocks)

    n_val_target = max(1, int(len(chip_names) * val_ratio)) if chip_names else 0

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        (val if len(val) < n_val_target else train).extend(blocks_map[block])
    return sorted(train), sorted(val)


CHIP_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def list_chip_paths(root: str | Path) -> list[Path]:
    """RGB chips in a folder. Training and evaluation read any of them (the scale comes from the OAM
    name), but turning GeoJSON labels into pixel boxes needs georeferenced GeoTIFFs."""
    folder = Path(root)
    return sorted(p for pattern in CHIP_PATTERNS for p in folder.glob(pattern) if p.is_file())


def _find_labels_geojson(labels_dir: str | Path) -> Path:
    root = Path(labels_dir)
    if root.is_file() and root.suffix.lower() == ".geojson":
        return root
    matches = sorted(root.glob("*.geojson"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .geojson in {root}, found {len(matches)}")
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
    out: list[tuple[float, float]] = []

    def _walk(n: Any) -> None:
        if not isinstance(n, (list, tuple)):
            return
        if len(n) >= 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in n):
            out.append((float(n[0]), float(n[1])))
            return
        for child in n:
            _walk(child)

    _walk(node)
    return out


def load_label_bounds(labels_geojson: str | Path) -> tuple[np.ndarray, str]:
    """Crown bounding boxes `(N, 4)` as `(minx, miny, maxx, maxy)` in the file CRS, plus that CRS."""
    fc = json.loads(Path(labels_geojson).read_text(encoding="utf-8"))
    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
        raise ValueError("labels file must be a GeoJSON FeatureCollection")
    features = fc.get("features", [])
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection 'features' must be a list")
    bounds: list[tuple[float, float, float, float]] = []
    for feat in features:
        geom = feat.get("geometry") if isinstance(feat, dict) else None
        coords = _iter_xy_coords(geom.get("coordinates")) if isinstance(geom, dict) else []
        if coords:
            xs, ys = zip(*coords)
            bounds.append((min(xs), min(ys), max(xs), max(ys)))
    return np.asarray(bounds, dtype=np.float64).reshape(-1, 4), _parse_geojson_crs(fc)


def _project_bounds(bounds: np.ndarray, src_crs: Any, dst_crs: Any) -> np.ndarray:
    """Reproject `(N, 4)` bounds by projecting all four corners and re-bounding."""
    if len(bounds) == 0 or src_crs is None or dst_crs is None or str(src_crs) == str(dst_crs):
        return bounds
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = np.stack([bounds[:, 0], bounds[:, 0], bounds[:, 2], bounds[:, 2]], axis=1)
    ys = np.stack([bounds[:, 1], bounds[:, 3], bounds[:, 1], bounds[:, 3]], axis=1)
    tx, ty = transformer.transform(xs, ys)
    return np.stack([tx.min(axis=1), ty.min(axis=1), tx.max(axis=1), ty.max(axis=1)], axis=1)


def boxes_for_chip(
    chip_path: str | Path,
    *,
    label_bounds: np.ndarray,
    labels_crs: Any,
    class_id: int = 0,
) -> list[dict[str, float | int]]:
    """Label bounds -> XYXY boxes in chip pixels, clipped to the chip; boxes off the chip are dropped."""
    import rasterio

    with rasterio.open(chip_path) as src:
        width, height, transform, crs = int(src.width), int(src.height), src.transform, src.crs
    if crs is None:
        raise ValueError(f"{chip_path} has no CRS; GeoJSON labels need a georeferenced GeoTIFF chip")
    if len(label_bounds) == 0:
        return []

    b = _project_bounds(label_bounds, labels_crs, crs)
    x = np.stack([b[:, 0], b[:, 0], b[:, 2], b[:, 2]], axis=1)
    y = np.stack([b[:, 1], b[:, 3], b[:, 1], b[:, 3]], axis=1)
    inv = ~transform
    # Explicit affine maths (same reason as georef.py): works across `affine` versions.
    cols = inv.a * x + inv.b * y + inv.c
    rows = inv.d * x + inv.e * y + inv.f
    x1, x2 = np.clip(cols.min(axis=1), 0, width), np.clip(cols.max(axis=1), 0, width)
    y1, y2 = np.clip(rows.min(axis=1), 0, height), np.clip(rows.max(axis=1), 0, height)
    keep = (x2 > x1) & (y2 > y1)
    return [
        {"x1": float(a), "y1": float(b_), "x2": float(c), "y2": float(d), "class_id": int(class_id)}
        for a, b_, c, d in zip(x1[keep], y1[keep], x2[keep], y2[keep])
    ]


def labels_for_chips(
    chips_dir: str | Path, labels_dir: str | Path, *, class_id: int = 0
) -> dict[str, list[dict[str, float | int]]]:
    """Pixel boxes of every chip in `chips_dir` from one GeoJSON of box or polygon labels. Nothing is copied.

    A pipeline step can call this on the chips and labels it was given, so no prepared folder has to be shared
    between isolated steps. Chips without a label get an empty list (negative examples).
    """
    bounds, crs = load_label_bounds(_find_labels_geojson(labels_dir))
    return {
        p.name: boxes_for_chip(p, label_bounds=bounds, labels_crs=crs, class_id=class_id)
        for p in list_chip_paths(chips_dir)
    }
