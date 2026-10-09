"""Window/model pixels -> chip pixels -> EPSG:4326 polygons (no ONNX layout knowledge)."""

from typing import Any

from pyproj import Transformer

from .preprocess import WindowMeta


def window_to_chip(det: dict[str, float | int], window: WindowMeta) -> tuple[float, float, float, float]:
    """Undo the model resize, then add the window offset -> source chip pixels."""
    return (
        det["x1"] / window.scale_x + window.x,
        det["y1"] / window.scale_y + window.y,
        det["x2"] / window.scale_x + window.x,
        det["y2"] / window.scale_y + window.y,
    )


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
