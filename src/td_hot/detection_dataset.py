"""Chips + labels as a torch dataset for RetinaNet fine-tuning (needs the `train` extra; import explicitly)."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import flip_chip_and_boxes, list_chip_paths, load_labels, scale_chip_and_boxes
from .params import PreprocessParams
from .preprocess import read_chip


class DetectionDataset(torch.utils.data.Dataset):
    """Chips + `labels.json` (from `build_chip_labels`) as torchvision targets.

    Each chip is scaled exactly as at serving time and its boxes are scaled with it (`scale_chip_and_boxes`).
    With `augment`, a chip is mirrored left-right with probability 0.5 (`flip_chip_and_boxes`, DeepForest's training
    augmentation). The model's own transform then only normalises and pads.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels: str | Path | Mapping[str, list[dict[str, Any]]],
        pre: PreprocessParams,
        *,
        names: Sequence[str] | None = None,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.pre = pre
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.paths = list_chip_paths(images_dir)
        if names is not None:
            wanted = set(names)
            self.paths = [path for path in self.paths if path.name in wanted]
            missing = wanted - {path.name for path in self.paths}
            if missing:
                raise ValueError(
                    f"{len(missing)} named chips not found in {images_dir}, e.g. {sorted(missing)[:3]}"
                )
        if not self.paths:
            raise ValueError(f"No chips found in {images_dir}")
        self.labels = load_labels(labels)

    def __len__(self) -> int:
        return len(self.paths)

    def _boxes(self, name: str) -> np.ndarray:
        """The chip's label boxes as `(N, 4)` float32 XYXY in chip pixels; `(0, 4)` for an unlabelled chip."""
        return np.asarray(
            [[box["x1"], box["y1"], box["x2"], box["y2"]] for box in self.labels.get(name, [])],
            dtype=np.float32,
        ).reshape(-1, 4)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        path = self.paths[idx]
        rgb, _, _ = read_chip(path)
        image, boxes = scale_chip_and_boxes(rgb, self._boxes(path.name), self.pre, path.name)
        if self.augment and self.rng.random() < 0.5:
            image, boxes = flip_chip_and_boxes(image, boxes)
        # One class (`Tree`, id 0); RetinaNet has no background label. Empty box lists are negatives.
        return torch.from_numpy(image), {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.zeros(len(boxes), dtype=torch.int64),
        }
