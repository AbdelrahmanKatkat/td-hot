"""Dataset helpers for td-hot (numpy/rasterio/pyproj only; safe to import at serving time).

- `spatial_split`: deterministic block split of OAM chips into train/val.
- `prepare_detection_dataset_from_geojson`: crown polygons (GeoJSON) -> per-chip pixel boxes.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .params import SplitParams

OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


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


def list_chip_paths(root: str | Path) -> list[Path]:
    """GeoTIFF chips only; training reads georeferenced `.tif`/`.tiff` chips."""
    folder = Path(root)
    return sorted(p for p in [*folder.glob("*.tif"), *folder.glob("*.tiff")] if p.is_file())


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


def prepare_detection_dataset_from_geojson(
    *,
    chips_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    split_cfg: SplitParams,
    class_id: int = 0,
) -> dict[str, Any]:
    """Split OAM chips spatially and write per-split chips plus pixel-space box labels.

    Output layout:
    - `<out_dir>/train/images/*.tif` and `<out_dir>/train/labels.json`
    - `<out_dir>/val/images/*.tif` and `<out_dir>/val/labels.json`

    `labels.json` maps chip filename -> list of `{x1, y1, x2, y2, class_id}` in chip pixels. Chips
    with no crown get an empty list (negative examples). `out_dir` must not already hold files, so a
    re-run with another seed cannot leave chips from the old split behind.
    """
    chips_root = Path(chips_dir)
    out_root = Path(out_dir)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"{out_root} is not empty; use a fresh output directory")

    label_bounds, labels_crs = load_label_bounds(_find_labels_geojson(labels_dir))
    chip_paths = list_chip_paths(chips_root)
    if len(chip_paths) < 2:
        raise ValueError("Need at least 2 chips for train/val split")

    train_names, val_names = spatial_split(
        [p.name for p in chip_paths],
        split_cfg.val_ratio,
        split_cfg.split_seed,
        block_size=split_cfg.block_size,
    )
    if not train_names or not val_names:
        raise ValueError("Spatial split left train or val empty; add chips or lower block_size/val_ratio")

    total_boxes = 0
    for split_name, names in (("train", train_names), ("val", val_names)):
        img_dir = out_root / split_name / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        labels_map: dict[str, list[dict[str, float | int]]] = {}
        for name in names:
            shutil.copy2(chips_root / name, img_dir / name)
            boxes = boxes_for_chip(
                chips_root / name, label_bounds=label_bounds, labels_crs=labels_crs, class_id=class_id
            )
            labels_map[name] = boxes
            total_boxes += len(boxes)
        (out_root / split_name / "labels.json").write_text(json.dumps(labels_map), encoding="utf-8")

    return {
        "strategy": "spatial",
        "val_ratio": split_cfg.val_ratio,
        "seed": split_cfg.split_seed,
        "block_size": split_cfg.block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "train_chip_names": train_names,
        "val_chip_names": val_names,
        "total_boxes": total_boxes,
    }
