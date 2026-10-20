"""Torchvision RetinaNet fine-tuning (needs the `train` extra; import explicitly, not from `td_hot`)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from .dataset import list_chip_paths
from .export import build_retinanet
from .params import TrainParams
from .preprocess import read_rgb_chip


class DetectionDataset(torch.utils.data.Dataset):
    """Chips + `labels.json` (from `prepare_detection_dataset_from_geojson`) as torchvision targets.

    Images are `(3, H, W)` float in [0, 1]; torchvision's own transform then does the same
    normalise / resize-to-800 / pad that `preprocess.py` replicates for inference.
    """

    def __init__(self, images_dir: str | Path, labels_json: str | Path) -> None:
        self.paths = list_chip_paths(images_dir)
        if not self.paths:
            raise ValueError(f"No GeoTIFF chips found in {images_dir}")
        labels = json.loads(Path(labels_json).read_text(encoding="utf-8"))
        if not isinstance(labels, dict):
            raise ValueError("labels.json must map chip filename -> list of boxes")
        self.labels: dict[str, list[dict[str, Any]]] = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        path = self.paths[idx]
        rgb, _, _ = read_rgb_chip(path)
        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        boxes = np.asarray(
            [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in self.labels.get(path.name, [])], dtype=np.float32
        ).reshape(-1, 4)
        # One class (`Tree`, id 0); RetinaNet has no background label. Empty box lists are negatives.
        return image, {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.zeros(len(boxes), dtype=torch.int64),
        }


@dataclass(frozen=True)
class TrainResult:
    model: Any  # torchvision RetinaNet, best weights, CPU, eval mode
    history: dict[str, list[float]]
    best_epoch: int
    best_loss: float  # validation loss if a val set was given, otherwise train loss


def _collate(batch: list[tuple[Any, Any]]) -> tuple[list[Any], list[Any]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def _resolve_device(requested: str) -> str:
    raw = requested.strip().lower()
    if raw == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if raw in {"auto", "cuda", "gpu"}:
        return "cuda"
    return f"cuda:{raw}" if raw.isdigit() else raw


def _set_train_mode(model: torch.nn.Module) -> None:
    """Train mode, but BatchNorm keeps its pretrained running stats.

    The NEON checkpoint uses plain BatchNorm2d; with batch sizes of 1-4 its statistics would be
    re-estimated from a handful of chips (and drift again during the validation pass).
    torchvision itself uses frozen BatchNorm when fine-tuning pretrained detectors.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()


def _run_epoch(model: Any, loader: Any, device: str, optimizer: Any = None) -> float:
    """Mean total loss over `loader`; trains when `optimizer` is given, else only measures."""
    _set_train_mode(model)
    losses: list[float] = []
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_retinanet(
    *,
    train_images_dir: str | Path,
    train_labels_json: str | Path,
    val_images_dir: str | Path | None,
    val_labels_json: str | Path | None,
    cfg: TrainParams,
    checkpoint_path: str | Path | None = None,
) -> TrainResult:
    """Fine-tune the one-class RetinaNet on chip-pixel boxes; keep the best epoch.

    `checkpoint_path` is the DeepForest `NEON.pt` state dict (loaded strict). Without it the model
    starts from random weights. Export the result with `td_hot.export.export_onnx_bytes`.
    """
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    train_ds: Any = DetectionDataset(train_images_dir, train_labels_json)
    if cfg.sample_fraction < 1.0:
        n = max(1, round(len(train_ds) * cfg.sample_fraction))
        keep = np.random.default_rng(cfg.seed).permutation(len(train_ds))[:n].tolist()
        train_ds = torch.utils.data.Subset(train_ds, keep)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = None
    if val_images_dir is not None and val_labels_json is not None:
        val_loader = torch.utils.data.DataLoader(
            DetectionDataset(val_images_dir, val_labels_json),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

    model = build_retinanet(checkpoint_path).to(device)
    if cfg.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state: dict[str, torch.Tensor] = {}
    best_epoch, best_loss, patience_left = 0, float("inf"), cfg.early_stop_patience
    for epoch in range(cfg.epochs):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        history["train_loss"].append(train_loss)
        monitored = train_loss
        if val_loader is not None:
            monitored = _run_epoch(model, val_loader, device)
            history["val_loss"].append(monitored)

        if monitored < best_loss:
            best_epoch, best_loss, patience_left = epoch, monitored, cfg.early_stop_patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if not best_state:
        raise RuntimeError(f"training produced no finite loss (last: {monitored})")
    model.load_state_dict(best_state, strict=True)
    return TrainResult(
        model=model.to("cpu").eval(), history=history, best_epoch=best_epoch, best_loss=float(best_loss)
    )
