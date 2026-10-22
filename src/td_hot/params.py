"""Constants, defaults and dict -> dataclass parsing for `td-hot`.

Parsers tolerate `None` / missing keys. Defaults mirror DeepForest 2.1.0 (`conf/config.yaml`);
the model contract they belong to is documented in docs/Theory.md sections 2-6.

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
# Refuse inputs that would be upscaled beyond this (a 256 px chip would exceed 4096 px, i.e. a zoom
# that is far too coarse for this model). Zoom 18 needs 10.6 (latitude 28) to 11.9 (equator), so it is
# accepted, but that scale is far outside what the model was trained or measured on (about 3x at zoom 20).
MAX_SCALE = 16.0
# The longest side of a scaled chip never exceeds this many pixels (`PreprocessParams.max_input_px`). Without it
# the input size follows the zoom: a 256 px chip at zoom 18 (latitude 28) becomes 2,700 px and a CPU pod with a
# few GiB is OOM-killed. 1024 px leaves zoom 20 untouched (about 760 px) and bounds the memory of coarser zooms.
MAX_INPUT_PX = 1024

# ResNet-50 stages of the encoder, shallow to deep; `unfreeze_from` picks the first one that trains.
ENCODER_STAGES = ("layer1", "layer2", "layer3", "layer4")


@dataclass(frozen=True)
class SplitParams:
    val_ratio: float = 0.2
    split_seed: int = 42
    block_size: int = 4


@dataclass(frozen=True)
class PreprocessParams:
    """How a chip is scaled before the model. Training and serving must use the same value."""

    model_gsd_m: float = MODEL_GSD_M  # ground size of one model-input pixel, metres
    max_input_px: int = MAX_INPUT_PX  # longest side of the scaled chip; a coarser zoom is upscaled less


@dataclass(frozen=True)
class PostprocessParams:
    """Thresholds applied outside the ONNX graph (which returns candidates before NMS)."""

    confidence_threshold: float = 0.1
    nms_threshold: float = 0.05  # RetinaNet.nms_thresh, across all candidates of the chip


@dataclass(frozen=True)
class TrainParams:
    """Torchvision RetinaNet fine-tuning knobs; the defaults are the recipe of the fAIr `deepforest_tree` pack.

    SGD with momentum 0.9, gradient clipping and a cosine schedule. The ResNet encoder is frozen by default
    (BatchNorm always is), so fine-tuning only moves the FPN and the heads: this limits catastrophic forgetting
    of what the checkpoint learned on NEON. Set `freeze_encoder=False` only with plenty of target data.
    """

    epochs: int = 5
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0  # 0 disables clipping
    scheduler: str = (
        "cosine"  # "cosine" | "none"; cosine decays to 0 over all steps of the run (stepped per batch)
    )
    warmup_steps: int = 0  # linear learning-rate ramp over the first batches (0 = none)
    freeze_encoder: bool = True
    unfreeze_from: str = (
        "none"  # with freeze_encoder: still train ResNet stages from here on ("layer4", "layer3", ...)
    )
    encoder_lr_factor: float = (
        1.0  # learning-rate multiplier for the encoder weights that do train (1.0 = same as the heads)
    )
    early_stop_patience: int = (
        5  # epochs without a better loss before stopping (>= epochs: never stops early)
    )
    sample_fraction: float = 1.0  # one fixed random subset of the training tiles for the whole run
    epoch_fraction: float = (
        1.0  # each epoch trains on a fresh random fraction of the tiles (1.0 = all of them)
    )
    device: str = "auto"  # cuda if visible, else cpu (training is meant to run on a GPU)
    seed: int = 42  # shuffling, augmentation and `sample_fraction` subsampling


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
    """`model_gsd_m`: metres per model-input pixel (default 0.05, the NEON training scale).

    `max_input_px`: cap on the longest side of the scaled chip (default 1024), which bounds memory at coarse zooms.
    """
    p = params or {}
    model_gsd_m = _as_float(_get(p, ("model_gsd_m",), PreprocessParams.model_gsd_m))
    if not 0.005 <= model_gsd_m <= 1.0:
        raise ValueError(f"model_gsd_m must be in [0.005, 1.0] metres, got {model_gsd_m}")
    max_input_px = _as_int(_get(p, ("max_input_px",), PreprocessParams.max_input_px))
    if not 64 <= max_input_px <= 8192:
        raise ValueError(f"max_input_px must be in [64, 8192] pixels, got {max_input_px}")
    return PreprocessParams(model_gsd_m=model_gsd_m, max_input_px=max_input_px)


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
    epoch_fraction = _as_float(_get(p, ("epoch_fraction",), TrainParams.epoch_fraction))
    warmup_steps = _as_int(_get(p, ("warmup_steps",), TrainParams.warmup_steps))
    max_grad_norm = _as_float(_get(p, ("max_grad_norm",), TrainParams.max_grad_norm))
    scheduler = str(_get(p, ("scheduler",), TrainParams.scheduler)).strip().lower()
    unfreeze_from = str(_get(p, ("unfreeze_from",), TrainParams.unfreeze_from)).strip().lower()
    encoder_lr_factor = _as_float(_get(p, ("encoder_lr_factor",), TrainParams.encoder_lr_factor))
    freeze_encoder = _as_bool(_get(p, ("freeze_encoder", "freeze_backbone"), TrainParams.freeze_encoder))
    for name, ok, value in (
        ("epochs", epochs >= 1, epochs),
        ("batch_size", batch_size >= 1, batch_size),
        ("learning_rate", learning_rate > 0, learning_rate),
        ("weight_decay", weight_decay >= 0, weight_decay),
        ("early_stop_patience", patience >= 1, patience),
        ("sample_fraction", 0.0 < sample_fraction <= 1.0, sample_fraction),
        ("epoch_fraction", 0.0 < epoch_fraction <= 1.0, epoch_fraction),
        ("warmup_steps", warmup_steps >= 0, warmup_steps),
        ("max_grad_norm", max_grad_norm >= 0, max_grad_norm),
        ("scheduler", scheduler in ("cosine", "none"), scheduler),
        ("unfreeze_from", unfreeze_from in ENCODER_STAGES + ("none",), unfreeze_from),
        ("encoder_lr_factor", 0.0 < encoder_lr_factor <= 1.0, encoder_lr_factor),
    ):
        if not ok:
            raise ValueError(f"{name} out of range, got {value}")
    if unfreeze_from != "none" and not freeze_encoder:
        raise ValueError(
            "unfreeze_from needs freeze_encoder=True (with freeze_encoder=False the whole encoder trains)"
        )
    return TrainParams(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        early_stop_patience=patience,
        freeze_encoder=freeze_encoder,
        unfreeze_from=unfreeze_from,
        encoder_lr_factor=encoder_lr_factor,
        sample_fraction=sample_fraction,
        epoch_fraction=epoch_fraction,
        device=str(_get(p, ("device",), TrainParams.device)),
        seed=_as_int(_get(p, ("seed",), TrainParams.seed)),
    )
