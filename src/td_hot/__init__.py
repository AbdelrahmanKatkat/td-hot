"""`td-hot`: HOT fAIr DeepForest tree-crown detector helpers."""

from .dataset import prepare_detection_dataset_from_geojson, spatial_split
from .decode import decode_retinanet_output, merge_window_detections, nms_xyxy
from .params import (
    InferenceParams,
    PostprocessParams,
    PreprocessParams,
    SplitParams,
    TrainParams,
    parse_inference_params,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
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
    "TrainParams",
    "WindowMeta",
    "decode_retinanet_output",
    "detections_to_feature_collection",
    "merge_window_detections",
    "nms_xyxy",
    "parse_inference_params",
    "parse_postprocess_params",
    "parse_preprocess_params",
    "parse_split_params",
    "parse_train_params",
    "predict_chip",
    "predict_session",
    "prepare_detection_dataset_from_geojson",
    "preprocess_chip",
    "sliding_windows",
    "spatial_split",
]
# Optional (train extra, import explicitly): td_hot.export, td_hot.train, td_hot.evaluate
