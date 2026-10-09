"""Chip -> sliding windows -> ONNX-ready tensors (mirrors DeepForest + torchvision transform)."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .params import (
    IMAGE_MEAN,
    IMAGE_STD,
    MAX_SIZE,
    MIN_SIZE,
    SIZE_DIVISIBLE,
    PreprocessParams,
)


@dataclass(frozen=True)
class WindowMeta:
    """One window: position in the chip and the resize applied before the model."""

    x: int
    y: int
    width: int
    height: int
    resized_width: int  # pixels after the model resize, before padding
    resized_height: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.height


@dataclass(frozen=True)
class PreprocessMeta:
    """Everything needed to map detections back to chip pixels and then to the map."""

    width: int
    height: int
    windows: tuple[WindowMeta, ...]
    transform: Any
    crs: Any


def sliding_windows(
    width: int, height: int, patch_size: int, patch_overlap: float
) -> list[tuple[int, int, int, int]]:
    """`(x, y, w, h)` windows, identical to `slidingwindow.generateForSize` (used by DeepForest).

    Window size is clipped to the image; the last row/column is shifted back to end on the
    image edge, so every window has the full window size (more overlap near the edge).
    """
    win_w = min(patch_size, width)
    win_h = min(patch_size, height)
    step_x = win_w - math.floor(win_w * patch_overlap)
    step_y = win_h - math.floor(win_h * patch_overlap)
    last_x = width - win_w
    last_y = height - win_h
    xs = list(range(0, last_x + 1, step_x))
    ys = list(range(0, last_y + 1, step_y))
    if xs[-1] != last_x:
        xs.append(last_x)
    if ys[-1] != last_y:
        ys.append(last_y)
    return [(x, y, win_w, win_h) for x in xs for y in ys]


def read_rgb_chip(image_path: str | Path) -> tuple[np.ndarray, Any, Any]:
    """Read bands 1-3 as uint8 HWC RGB plus the chip's affine transform and CRS."""
    import rasterio

    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError(f"{image_path}: expected >= 3 bands (RGB), found {src.count}")
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        return np.clip(rgb, 0, 255).astype(np.uint8), src.transform, src.crs


def preprocess_window(rgb_hwc: np.ndarray) -> tuple[np.ndarray, int, int]:
    """RGB uint8 window -> `(1, 3, H', W')` float32 tensor, plus the resized (width, height).

    Replicates torchvision `GeneralizedRCNNTransform`: ImageNet mean/std on [0, 1] input,
    bilinear resize so the short side is MIN_SIZE (long side capped at MAX_SIZE), then
    zero-pad bottom/right to a multiple of 32.
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB array, got shape {rgb_hwc.shape}")
    h, w, _ = rgb_hwc.shape
    scale = min(MIN_SIZE / min(h, w), MAX_SIZE / max(h, w))
    new_h, new_w = math.floor(h * scale), math.floor(w * scale)

    norm = (rgb_hwc.astype(np.float32) / 255.0 - np.asarray(IMAGE_MEAN, np.float32)) / np.asarray(
        IMAGE_STD, np.float32
    )
    pad_h = math.ceil(new_h / SIZE_DIVISIBLE) * SIZE_DIVISIBLE
    pad_w = math.ceil(new_w / SIZE_DIVISIBLE) * SIZE_DIVISIBLE
    out = np.zeros((1, 3, pad_h, pad_w), dtype=np.float32)
    for c in range(3):
        resized = Image.fromarray(norm[:, :, c], mode="F").resize((new_w, new_h), Image.Resampling.BILINEAR)
        out[0, c, :new_h, :new_w] = np.asarray(resized, dtype=np.float32)
    return out, new_w, new_h


def preprocess_chip(
    image_path: str | Path, cfg: PreprocessParams | None = None
) -> tuple[list[np.ndarray], PreprocessMeta]:
    """Read a chip, cut sliding windows and return one tensor per window plus metadata."""
    cfg = cfg or PreprocessParams()
    rgb, transform, crs = read_rgb_chip(image_path)
    return preprocess_array(rgb, cfg, transform=transform, crs=crs)


def preprocess_array(
    rgb_hwc: np.ndarray, cfg: PreprocessParams | None = None, *, transform: Any = None, crs: Any = None
) -> tuple[list[np.ndarray], PreprocessMeta]:
    cfg = cfg or PreprocessParams()
    height, width, _ = rgb_hwc.shape
    batches: list[np.ndarray] = []
    windows: list[WindowMeta] = []
    for x, y, w, h in sliding_windows(width, height, cfg.patch_size, cfg.patch_overlap):
        batch, new_w, new_h = preprocess_window(rgb_hwc[y : y + h, x : x + w])
        batches.append(batch)
        windows.append(WindowMeta(x=x, y=y, width=w, height=h, resized_width=new_w, resized_height=new_h))
    meta = PreprocessMeta(width=width, height=height, windows=tuple(windows), transform=transform, crs=crs)
    return batches, meta
