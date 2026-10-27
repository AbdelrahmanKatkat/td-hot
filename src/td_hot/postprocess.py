"""Chip-pixel detections -> GeoJSON FeatureCollection."""

from typing import Any

from .georef import clip_xyxy_to_image, xyxy_to_epsg4326_polygon
from .preprocess import PreprocessMeta

CLASS_NAMES = {0: "Tree"}  # HF config.json label_dict {"Tree": 0}


def detections_to_feature_collection(
    detections: list[dict[str, float | int]],
    meta: PreprocessMeta,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """`detections` are in source-chip pixels; output is EPSG:4326 box polygons."""
    features: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = clip_xyxy_to_image(
            det["x1"], det["y1"], det["x2"], det["y2"], meta.width, meta.height
        )
        if x2 <= x1 or y2 <= y1:
            continue
        ring = xyxy_to_epsg4326_polygon(x1, y1, x2, y2, meta.transform, meta.crs)
        properties: dict[str, Any] = {
            "confidence": round(float(det["confidence"]), 4),
            "class_id": int(det["class_id"]),
            "class_name": CLASS_NAMES.get(int(det["class_id"]), "Tree"),
        }
        if source_name:
            properties["source"] = source_name
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}
