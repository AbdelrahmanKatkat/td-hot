"""Constants, defaults and dict -> dataclass parsing for `td-hot`.

Parsers tolerate `None` / missing keys. Defaults mirror DeepForest 2.1.0 (`conf/config.yaml`);
the model contract they belong to is documented in theory.md sections 2-6.

Where each value lives (the "level 2" ONNX contract):
- inside the ONNX graph: `IMAGE_MEAN`/`IMAGE_STD` normalisation, padding, backbone + FPN + head,
  anchors, box decode, `TOPK_CANDIDATES` per level, clipping. Fixed by `export.build_retinanet`.
- outside, tunable: scale (`PreprocessParams`), confidence filter, NMS (`PostprocessParams`).
"""

from dataclasses import dataclass
from typing import Any, Mapping

# --- model constants (not stored in the checkpoint; `export.build_retinanet` is the single owner) ---
# ImageNet statistics, used unchanged by DeepForest (torchvision transform default).
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
TOPK_CANDIDATES = 1000  # per FPN level, before NMS (inside the graph)
DETECTIONS_PER_IMG = 300  # after NMS (outside the graph)

# --- scale policy ---
# NEON training: 10 cm/px imagery cut into 400 px windows and upscaled to 800 px by torchvision,
# so one model pixel covers 0.10 m * 400 / 800 = 0.05 m of ground.
MODEL_GSD_M = 0.05
# Refuse inputs that would be upscaled beyond this (a 256 px chip would exceed 2048 px, i.e. a zoom
# that is far too coarse for this model). Not measured above 15.6 cm/px, see theory.md section 7.1.
MAX_SCALE = 8.0


@dataclass(frozen=True)
class SplitParams:
    val_ratio: float = 0.2
    split_seed: int = 42
    block_size: int = 4


@dataclass(frozen=True)
class PreprocessParams:
    """How a chip is scaled before the model. Training and serving must use the same value."""

    model_gsd_m: float = MODEL_GSD_M  # ground size of one model-input pixel, metres


@dataclass(frozen=True)
class PostprocessParams:
    """Thresholds applied outside the ONNX graph (which returns candidates before NMS)."""

    confidence_threshold: float = 0.1
    nms_threshold: float = 0.05  # RetinaNet.nms_thresh, across all candidates of the chip


@dataclass(frozen=True)
class TrainParams:
    """Torchvision RetinaNet fine-tuning knobs (kept minimal and fAIr-friendly)."""

    epochs: int = 5
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    early_stop_patience: int = 5
    freeze_backbone: bool = False
    sample_fraction: float = 1.0
    device: str = "auto"  # cuda if visible, else cpu
    seed: int = 42  # shuffling and `sample_fraction` subsampling


def _get(d: Mapping[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    """First present non-None key wins (canonical key first, then aliases)."""
    for key in keys:
        if d.get(key) is not None:
            return d[key]
    return default


def _as_int(v: Any) -> int:
    if isinstance(v, bool):
        raise ValueError("bool is not a valid int here")
    return int(v)


def _as_float(v: Any) -> float:
    if isinstance(v, bool):
        raise ValueError("bool is not a valid float here")
    return float(v)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        raw = v.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected bool-like value, got {v!r}")


def _check_unit(name: str, v: float, *, low_open: bool = False) -> float:
    ok = (0.0 < v <= 1.0) if low_open else (0.0 <= v <= 1.0)
    if not ok:
        raise ValueError(f"{name} must be in {'(0, 1]' if low_open else '[0, 1]'}, got {v}")
    return v


def parse_split_params(hyperparameters: Mapping[str, Any] | None) -> SplitParams:
    hp = hyperparameters or {}
    val_ratio = _check_unit(
        "val_ratio", _as_float(_get(hp, ("val_ratio",), SplitParams.val_ratio)), low_open=True
    )
    block_size = _as_int(_get(hp, ("block_size",), SplitParams.block_size))
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    return SplitParams(
        val_ratio=val_ratio,
        split_seed=_as_int(_get(hp, ("split_seed",), SplitParams.split_seed)),
        block_size=block_size,
    )


def parse_preprocess_params(params: Mapping[str, Any] | None) -> PreprocessParams:
    """`model_gsd_m`: metres of ground per model-input pixel (default 0.05, the NEON training scale)."""
    p = params or {}
    model_gsd_m = _as_float(_get(p, ("model_gsd_m",), PreprocessParams.model_gsd_m))
    if not 0.005 <= model_gsd_m <= 1.0:
        raise ValueError(f"model_gsd_m must be in [0.005, 1.0] metres, got {model_gsd_m}")
    return PreprocessParams(model_gsd_m=model_gsd_m)


def parse_postprocess_params(params: Mapping[str, Any] | None) -> PostprocessParams:
    """`confidence_threshold` is canonical; `score_thresh` (DeepForest name) is an alias."""
    p = params or {}
    return PostprocessParams(
        confidence_threshold=_check_unit(
            "confidence_threshold",
            _as_float(
                _get(p, ("confidence_threshold", "score_thresh"), PostprocessParams.confidence_threshold)
            ),
        ),
        nms_threshold=_check_unit(
            "nms_threshold",
            _as_float(_get(p, ("nms_threshold", "nms_thresh"), PostprocessParams.nms_threshold)),
        ),
    )


def parse_train_params(params: Mapping[str, Any] | None) -> TrainParams:
    """Parse training knobs from a flat dict (fAIr strips the `training.` prefix)."""
    p = params or {}
    epochs = _as_int(_get(p, ("epochs",), TrainParams.epochs))
    batch_size = _as_int(_get(p, ("batch_size",), TrainParams.batch_size))
    learning_rate = _as_float(_get(p, ("learning_rate", "lr"), TrainParams.learning_rate))
    weight_decay = _as_float(_get(p, ("weight_decay",), TrainParams.weight_decay))
    patience = _as_int(_get(p, ("early_stop_patience",), TrainParams.early_stop_patience))
    sample_fraction = _as_float(_get(p, ("sample_fraction",), TrainParams.sample_fraction))
    for name, ok, value in (
        ("epochs", epochs >= 1, epochs),
        ("batch_size", batch_size >= 1, batch_size),
        ("learning_rate", learning_rate > 0, learning_rate),
        ("weight_decay", weight_decay >= 0, weight_decay),
        ("early_stop_patience", patience >= 1, patience),
        ("sample_fraction", 0.0 < sample_fraction <= 1.0, sample_fraction),
    ):
        if not ok:
            raise ValueError(f"{name} out of range, got {value}")
    return TrainParams(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stop_patience=patience,
        freeze_backbone=_as_bool(_get(p, ("freeze_backbone",), TrainParams.freeze_backbone)),
        sample_fraction=sample_fraction,
        device=str(_get(p, ("device",), TrainParams.device)),
        seed=_as_int(_get(p, ("seed",), TrainParams.seed)),
    )
