"""`td-hot`: HOT fAIr DeepForest tree-crown detector helpers."""

from .dataset import build_chip_labels, spatial_split
from .decode import decode_detections, nms
from .params import (
    PostprocessParams,
    PreprocessParams,
    SplitParams,
    TrainParams,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)
from .postprocess import detections_to_feature_collection
from .preprocess import PreprocessMeta, preprocess_chip, preprocess_rgb
from .serve import predict_chip, predict_session
from .tiles import chip_scale, tile_gsd_m

__all__ = [
    # parameters
    "PreprocessParams",
    "PostprocessParams",
    "SplitParams",
    "TrainParams",
    "parse_preprocess_params",
    "parse_postprocess_params",
    "parse_split_params",
    "parse_train_params",
    # tiles and dataset
    "tile_gsd_m",
    "chip_scale",
    "build_chip_labels",
    "spatial_split",
    # preprocess -> decode -> postprocess
    "PreprocessMeta",
    "preprocess_rgb",
    "preprocess_chip",
    "nms",
    "decode_detections",
    "detections_to_feature_collection",
    # inference
    "predict_chip",
    "predict_session",
]
# Optional (train extra, import explicitly): td_hot.export, td_hot.detection_dataset, td_hot.train, td_hot.evaluate
