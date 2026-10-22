"""`predict_session`: directory of OAM chips -> scale -> ONNX (level 2) -> NMS -> GeoJSON."""

import logging
import time
from pathlib import Path
from typing import Any

from .decode import decode_detections
from .georef import detections_to_chip
from .params import parse_postprocess_params, parse_preprocess_params
from .postprocess import detections_to_feature_collection
from .preprocess import PreprocessMeta, preprocess_chip

log = logging.getLogger(__name__)


def iter_image_paths(input_dir: str | Path) -> list[Path]:
    """List chip files in a directory (png/tif/jpg)."""
    root = Path(input_dir)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    files = sorted(p for pattern in patterns for p in root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input images found in {root}")
    return files


def detect_chip(
    session: Any, image_path: str | Path, params: dict[str, Any] | None
) -> tuple[list[dict[str, float | int]], PreprocessMeta]:
    """One chip through the ONNX session: detections in source-chip pixels, plus the metadata to georeference them."""
    pre_cfg = parse_preprocess_params(params)
    post_cfg = parse_postprocess_params(params)

    image, meta = preprocess_chip(image_path, pre_cfg)
    log.debug(
        "%s: scale %.3f -> input %dx%d",
        Path(image_path).name,
        meta.scale_x,
        meta.resized_width,
        meta.resized_height,
    )
    boxes, scores, labels = session.run(None, {session.get_inputs()[0].name: image})[:3]
    dets = decode_detections(boxes, scores, labels, post_cfg)
    return detections_to_chip(dets, meta), meta


def predict_chip(session: Any, image_path: str | Path, params: dict[str, Any] | None) -> dict[str, Any]:
    """Predict one chip (a single model pass) and return its FeatureCollection."""
    dets, meta = detect_chip(session, image_path, params)
    return detections_to_feature_collection(dets, meta, source_name=Path(image_path).name)


def predict_session(session: Any, input_images: str | Path, params: dict[str, Any] | None) -> dict[str, Any]:
    """Run ONNX prediction for all chips in a directory."""
    features: list[dict[str, Any]] = []
    started = time.time()
    paths = iter_image_paths(input_images)
    for img_path in paths:
        features.extend(predict_chip(session, img_path, params)["features"])
    log.info(
        "predict_session: %d chips, %d detections in %.1fs", len(paths), len(features), time.time() - started
    )
    return {"type": "FeatureCollection", "features": features}
