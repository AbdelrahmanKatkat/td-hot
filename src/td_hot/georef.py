"""Model pixels -> chip pixels -> EPSG:4326 polygons (no ONNX layout knowledge)."""

from typing import Any

from pyproj import Transformer

from .preprocess import PreprocessMeta


def resized_to_chip(det: dict[str, float | int], meta: PreprocessMeta) -> tuple[float, float, float, float]:
    """Undo the model resize -> source chip pixels."""
    return (
        det["x1"] / meta.scale_x,
        det["y1"] / meta.scale_y,
        det["x2"] / meta.scale_x,
        det["y2"] / meta.scale_y,
    )


def detections_to_chip(
    dets: list[dict[str, float | int]], meta: PreprocessMeta
) -> list[dict[str, float | int]]:
    """Detections in model-input pixels -> the same detections in source-chip pixels."""
    out = []
    for det in dets:
        x1, y1, x2, y2 = resized_to_chip(det, meta)
        out.append({**det, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def clip_xyxy_to_image(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def xyxy_to_epsg4326_polygon(
    x1: float, y1: float, x2: float, y2: float, transform: Any, crs: Any
) -> list[list[float]]:
    """One chip-pixel box -> closed EPSG:4326 ring `[[lon, lat], ...]`."""
    corners_pixel = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    # Explicit affine maths: works across `affine` versions (`*` is deprecated, `@` needs affine>=3).
    corners_crs = [
        (
            transform.a * col + transform.b * row + transform.c,
            transform.d * col + transform.e * row + transform.f,
        )
        for col, row in corners_pixel
    ]
    if crs is not None and str(crs) != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return [list(transformer.transform(cx, cy)) for cx, cy in corners_crs]
    return [list(c) for c in corners_crs]
