"""`predict_session`: directory of chips -> sliding windows -> ONNX -> GeoJSON."""

from pathlib import Path
from typing import Any

from .decode import decode_retinanet_output, merge_window_detections
from .georef import window_to_chip
from .params import parse_postprocess_params, parse_preprocess_params
from .postprocess import detections_to_feature_collection
from .preprocess import preprocess_chip


def iter_image_paths(input_dir: str | Path) -> list[Path]:
    """List chip files in a directory (png/tif/jpg)."""
    root = Path(input_dir)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    files = sorted(p for pattern in patterns for p in root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input images found in {root}")
    return files


def predict_chip(session: Any, image_path: str | Path, params: dict[str, Any] | None) -> dict[str, Any]:
    """Predict one chip (all windows) and return its FeatureCollection."""
    pre_cfg = parse_preprocess_params(params)
    post_cfg = parse_postprocess_params(params)
    input_name = session.get_inputs()[0].name

    batches, meta = preprocess_chip(image_path, pre_cfg)
    chip_dets: list[dict[str, float | int]] = []
    for batch, window in zip(batches, meta.windows):
        cls_logits, bbox_regression = session.run(None, {input_name: batch})[:2]
        dets = decode_retinanet_output(
            cls_logits,
            bbox_regression,
            input_hw=(batch.shape[2], batch.shape[3]),
            image_hw=(window.resized_height, window.resized_width),
            cfg=post_cfg,
        )
        for det in dets:
            x1, y1, x2, y2 = window_to_chip(det, window)
            chip_dets.append({**det, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    merged = merge_window_detections(chip_dets, post_cfg.iou_threshold)
    return detections_to_feature_collection(merged, meta, source_name=Path(image_path).name)


def predict_session(session: Any, input_images: str | Path, params: dict[str, Any] | None) -> dict[str, Any]:
    """Run ONNX prediction for all chips in a directory.

    Called from `models/<name>/pipeline.py::predict`. `session` is an onnxruntime
    InferenceSession on the exported RetinaNet (outputs: cls_logits, bbox_regression).
    """
    features: list[dict[str, Any]] = []
    for img_path in iter_image_paths(input_images):
        features.extend(predict_chip(session, img_path, params)["features"])
    return {"type": "FeatureCollection", "features": features}
