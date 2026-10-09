"""`td-hot`: HOT fAIr DeepForest tree-crown detector helpers."""

from .dataset import spatial_split
from .decode import decode_retinanet_output, merge_window_detections, nms_xyxy
from .params import (
    InferenceParams,
    PostprocessParams,
    PreprocessParams,
    SplitParams,
    parse_inference_params,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
)
from .postprocess import detections_to_feature_collection
from .preprocess import PreprocessMeta, WindowMeta, preprocess_chip, sliding_windows
from .serve import predict_chip, predict_session

__all__ = [
    "InferenceParams",
    "PostprocessParams",
    "PreprocessMeta",
    "PreprocessParams",
    "SplitParams",
    "WindowMeta",
    "decode_retinanet_output",
    "detections_to_feature_collection",
    "merge_window_detections",
    "nms_xyxy",
    "parse_inference_params",
    "parse_postprocess_params",
    "parse_preprocess_params",
    "parse_split_params",
    "predict_chip",
    "predict_session",
    "preprocess_chip",
    "sliding_windows",
    "spatial_split",
]
# Optional (train extra, import explicitly): td_hot.export.export_onnx_bytes
