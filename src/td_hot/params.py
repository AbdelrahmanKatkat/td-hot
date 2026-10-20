"""Defaults and dict -> dataclass parsing for `td-hot`.

Parsers tolerate `None` / missing keys. Defaults mirror DeepForest 2.1.0
(`conf/config.yaml`, `main.predict_tile`); see theory.md section 6.
"""

from dataclasses import dataclass
from typing import Any, Mapping

# torchvision GeneralizedRCNNTransform defaults, used unchanged by DeepForest (theory.md 2.2).
MIN_SIZE = 800
MAX_SIZE = 1333
SIZE_DIVISIBLE = 32
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitParams:
    val_ratio: float = 0.2
    split_seed: int = 42
    block_size: int = 4


@dataclass(frozen=True)
class PreprocessParams:
    """Sliding-window controls (applied to the source chip, before the model resize)."""

    patch_size: int = 400
    patch_overlap: float = 0.05


@dataclass(frozen=True)
class PostprocessParams:
    """Model-space decode and window-merge thresholds."""

    confidence_threshold: float = 0.1
    nms_threshold: float = 0.05  # per-window NMS (RetinaNet.nms_thresh)
    iou_threshold: float = 0.15  # cross-window NMS (predict_tile iou_threshold)


@dataclass(frozen=True)
class InferenceParams:
    patch_size: int = 400
    patch_overlap: float = 0.05
    confidence_threshold: float = 0.1
    nms_threshold: float = 0.05
    iou_threshold: float = 0.15


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
    p = params or {}
    patch_size = _as_int(_get(p, ("patch_size",), PreprocessParams.patch_size))
    patch_overlap = _as_float(_get(p, ("patch_overlap",), PreprocessParams.patch_overlap))
    if patch_size < 32:
        raise ValueError(f"patch_size must be >= 32, got {patch_size}")
    if not 0.0 <= patch_overlap < 1.0:
        raise ValueError(f"patch_overlap must be in [0, 1), got {patch_overlap}")
    return PreprocessParams(patch_size=patch_size, patch_overlap=patch_overlap)


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
        iou_threshold=_check_unit(
            "iou_threshold", _as_float(_get(p, ("iou_threshold",), PostprocessParams.iou_threshold))
        ),
    )


def parse_inference_params(params: Mapping[str, Any] | None) -> InferenceParams:
    pre = parse_preprocess_params(params)
    post = parse_postprocess_params(params)
    return InferenceParams(
        patch_size=pre.patch_size,
        patch_overlap=pre.patch_overlap,
        confidence_threshold=post.confidence_threshold,
        nms_threshold=post.nms_threshold,
        iou_threshold=post.iou_threshold,
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
