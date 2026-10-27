"""Torchvision RetinaNet fine-tuning (needs the `train` extra; import explicitly, not from `td_hot`).

The recipe is the one of the fAIr `deepforest_tree` pack: SGD with momentum, a cosine schedule (per batch, with an
optional linear warmup), gradient
clipping, horizontal-flip augmentation, a frozen encoder and always-frozen BatchNorm. Freezing the encoder keeps
what `NEON.pt` learned (catastrophic forgetting) while the FPN and the heads adapt to the new imagery.
"""

import logging
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from .export import build_retinanet
from .params import ENCODER_STAGES, PreprocessParams, TrainParams
from .detection_dataset import DetectionDataset

log = logging.getLogger(__name__)

LOG_LINES_PER_EPOCH = (
    50  # automatic step logging aims at this many lines per epoch, however large the dataset
)


@dataclass(frozen=True)
class TrainResult:
    model: Any  # torchvision RetinaNet, best weights, CPU, eval mode
    history: dict[str, list[float]]
    best_epoch: int
    best_loss: float  # validation loss if a val set was given, otherwise train loss


# Called after each epoch with (epoch starting at 1, train loss, validation loss or None).
EpochCallback = Callable[[int, float, "float | None"], None]


def _resolve_device(requested: str) -> str:
    raw = requested.strip().lower()
    if raw == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if raw in {"auto", "cuda", "gpu"}:
        return "cuda"
    return f"cuda:{raw}" if raw.isdigit() else raw


def _checkpoint_label(checkpoint: Any) -> str:
    return str(checkpoint) if checkpoint is None or isinstance(checkpoint, (str, Path)) else "<stream>"


def _ensure_visible_progress() -> None:
    """Make INFO progress lines appear even when the host process never configured logging.

    A bare container step has no root handler and a WARNING root level, so `log.info` would be silent while training
    runs. If nothing is configured, attach a stdout handler to this logger only; if handlers exist but filter out INFO
    at the logger level, enable INFO here. A host that configured logging itself (ZenML, uvicorn, `basicConfig`) is
    left alone.
    """
    if not log.isEnabledFor(logging.INFO):
        log.setLevel(logging.INFO)
    if not logging.getLogger().handlers and not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
        log.propagate = False


def _freeze(model: torch.nn.Module, *, freeze_encoder: bool, unfreeze_from: str = "none") -> None:
    """Freeze what must not move. BatchNorm affine parameters are always frozen.

    With `freeze_encoder` the whole ResNet encoder (`backbone.body`) is frozen except the stages from
    `unfreeze_from` on (for example `layer4` trains only the deepest stage, which holds the most
    domain-specific features); the FPN and the heads always train.
    """
    if freeze_encoder:
        body = model.backbone.body
        for param in body.parameters():
            param.requires_grad_(False)
        if unfreeze_from != "none":
            for stage in ENCODER_STAGES[ENCODER_STAGES.index(unfreeze_from) :]:
                for param in getattr(body, stage).parameters():
                    param.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            for param in module.parameters():
                param.requires_grad_(False)


def _optimizer_groups(model: torch.nn.Module, lr: float, encoder_lr_factor: float) -> list[dict[str, Any]]:
    """Parameter groups: neck and heads at `lr`, the encoder weights that train at `lr * encoder_lr_factor`."""
    encoder = [param for param in model.backbone.body.parameters() if param.requires_grad]
    encoder_ids = {id(param) for param in encoder}
    rest = [param for param in model.parameters() if param.requires_grad and id(param) not in encoder_ids]
    groups = [{"params": rest, "lr": lr}]
    if encoder:
        groups.append({"params": encoder, "lr": lr * encoder_lr_factor})
    return groups


def _lr_schedule(
    optimizer: Any, total_steps: int, warmup_steps: int, cosine: bool
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Per-batch learning-rate factor: a linear warmup, then cosine decay to 0 at `total_steps` (or flat)."""
    if not cosine and warmup_steps <= 0:
        return None

    def factor(step: int) -> float:
        warm = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        decay = 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)) if cosine else 1.0
        return warm * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _collate(batch: list[tuple[Any, Any]]) -> tuple[list[Any], list[Any]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def _epoch_loader(train_ds: Any, cfg: TrainParams, epoch: int) -> Any:
    """Shuffled loader for one epoch: all tiles, or a fresh random `cfg.epoch_fraction` of them (seeded per epoch)."""
    if cfg.epoch_fraction < 1.0:
        count = max(1, round(len(train_ds) * cfg.epoch_fraction))
        keep = np.random.default_rng(cfg.seed + epoch).permutation(len(train_ds))[:count].tolist()
        train_ds = torch.utils.data.Subset(train_ds, keep)
    return torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(cfg.seed + epoch),
    )


def _set_train_mode(model: torch.nn.Module) -> None:
    """Train mode, but BatchNorm keeps its pretrained running stats.

    The NEON checkpoint uses plain BatchNorm2d; with batch sizes of 1-4 its statistics would be
    re-estimated from a handful of chips (and drift again during the validation pass).
    torchvision itself uses frozen BatchNorm when fine-tuning pretrained detectors.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _run_epoch(
    model: Any,
    loader: Any,
    device: str,
    optimizer: Any = None,
    *,
    label: str = "",
    log_every: int = 0,
    max_grad_norm: float = 0.0,
    scheduler: Any = None,
) -> float:
    """Mean total loss over `loader`; trains when `optimizer` is given, else only measures.

    Logs the running loss (and its two parts) every `log_every` steps with an ETA; 0 chooses a step that gives about
    `LOG_LINES_PER_EPOCH` lines per epoch. A non-finite loss raises at once.
    """
    _set_train_mode(model)
    losses: list[float] = []
    part_losses: dict[str, list[float]] = {}
    start, steps = time.time(), len(loader)
    every = log_every or max(1, steps // LOG_LINES_PER_EPOCH)
    with torch.set_grad_enabled(optimizer is not None):
        for step, (images, targets) in enumerate(loader, 1):
            images = [image.to(device) for image in images]
            targets = [{name: tensor.to(device) for name, tensor in target.items()} for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{label} step {step}: non-finite loss; lower learning_rate or max_grad_norm"
                )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            losses.append(float(loss.detach().cpu()))
            for name, part in loss_dict.items():
                part_losses.setdefault(name, []).append(float(part.detach().cpu()))
            if step % every == 0 or step == steps:
                spent = time.time() - start
                recent = " ".join(
                    f"{name}={np.mean(values[-every:]):.4f}" for name, values in part_losses.items()
                )
                log.info(
                    "%s step %d/%d loss(last %d)=%.4f [%s] mean=%.4f %.2fs/step eta %s",
                    label, step, steps, every, np.mean(losses[-every:]), recent,
                    np.mean(losses), spent / step, _hms(spent / step * (steps - step)),
                )  # fmt: skip
    return float(np.mean(losses))


def train(
    *,
    train_images_dir: str | Path,
    train_labels: str | Path | Mapping[str, list[dict[str, Any]]],
    val_images_dir: str | Path | None,
    val_labels: str | Path | Mapping[str, list[dict[str, Any]]] | None,
    cfg: TrainParams,
    pre: PreprocessParams | None = None,
    checkpoint: str | Path | BinaryIO | None = None,
    log_every: int = 0,
    on_epoch_end: EpochCallback | None = None,
    max_val_chips: int | None = None,
    train_names: Sequence[str] | None = None,
    val_names: Sequence[str] | None = None,
    epoch_checkpoint_dir: str | Path | None = None,
) -> TrainResult:
    """Fine-tune the one-class RetinaNet on chip-pixel boxes; keep the best epoch.

    `checkpoint` is the DeepForest `NEON.pt` state dict, as a path or an open binary stream (loaded strict). Without
    it the model starts from random weights. `pre.model_gsd_m` must be the value used at serving time. Export the
    result with `td_hot.export.export_onnx_bytes`.

    Progress goes to the `td_hot.train` logger at INFO (and is made visible if the host configured nothing):
    a start summary, step lines (every `log_every` steps, or about 50 per epoch when 0), and one line per epoch.
    `on_epoch_end(epoch, train_loss, val_loss)` lets a pipeline step forward the numbers to MLflow or ZenML.
    `max_val_chips` limits the validation set to that many evenly spaced chips (quick pilots); by default every
    validation chip is used. `epoch_checkpoint_dir` saves the weights of every epoch as `epoch_NN.pt` (the best one
    by loss is returned either way), so epochs can be compared by F1 afterwards.

    Labels are a `labels.json` path or the mapping itself (`dataset.build_chip_labels`); `train_names` / `val_names`
    select chips of one folder, so a pipeline step can split the original chips without copying them.
    """
    _ensure_visible_progress()
    pre = pre or PreprocessParams()
    device = _resolve_device(cfg.device)
    if device == "cpu":
        log.warning("training on CPU: expect it to be slow; fAIr training is meant to run on a GPU")
    torch.manual_seed(cfg.seed)

    train_ds: Any = DetectionDataset(
        train_images_dir, train_labels, pre, names=train_names, augment=True, seed=cfg.seed
    )
    if cfg.sample_fraction < 1.0:
        count = max(1, round(len(train_ds) * cfg.sample_fraction))
        keep = np.random.default_rng(cfg.seed).permutation(len(train_ds))[:count].tolist()
        train_ds = torch.utils.data.Subset(train_ds, keep)
    steps_per_epoch = math.ceil(max(1, round(len(train_ds) * cfg.epoch_fraction)) / cfg.batch_size)
    val_loader = None
    if val_images_dir is not None and val_labels is not None:
        val_ds: Any = DetectionDataset(val_images_dir, val_labels, pre, names=val_names)
        if max_val_chips and max_val_chips < len(val_ds):
            stride = len(val_ds) / max_val_chips
            val_ds = torch.utils.data.Subset(val_ds, [int(i * stride) for i in range(max_val_chips)])
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

    model = build_retinanet(checkpoint).to(device)
    _freeze(model, freeze_encoder=cfg.freeze_encoder, unfreeze_from=cfg.unfreeze_from)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(
        _optimizer_groups(model, cfg.learning_rate, cfg.encoder_lr_factor), lr=cfg.learning_rate,
        momentum=0.9, weight_decay=cfg.weight_decay,
    )  # fmt: skip
    scheduler = _lr_schedule(
        optimizer, cfg.epochs * steps_per_epoch, cfg.warmup_steps, cosine=cfg.scheduler == "cosine"
    )
    log.info(
        "start: device=%s train_tiles=%d epoch_fraction=%g val_tiles=%s batch_size=%d steps/epoch=%d epochs=%d "
        "lr=%g scheduler=%s warmup_steps=%d max_grad_norm=%g freeze_encoder=%s unfreeze_from=%s encoder_lr_factor=%g params=%.1fM trainable=%.1fM model_gsd_m=%g seed=%d "
        "checkpoint=%s",
        device, len(train_ds), cfg.epoch_fraction, len(val_loader.dataset) if val_loader else "none",
        cfg.batch_size, steps_per_epoch, cfg.epochs, cfg.learning_rate, cfg.scheduler, cfg.warmup_steps,
        cfg.max_grad_norm, cfg.freeze_encoder,
        cfg.unfreeze_from, cfg.encoder_lr_factor,
        sum(param.numel() for param in model.parameters()) / 1e6, sum(param.numel() for param in trainable) / 1e6,
        pre.model_gsd_m, cfg.seed, _checkpoint_label(checkpoint),
    )  # fmt: skip

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state: dict[str, torch.Tensor] = {}
    best_epoch, best_loss, patience_left = 0, float("inf"), cfg.early_stop_patience
    started = time.time()
    for epoch in range(cfg.epochs):
        tag = f"epoch {epoch + 1}/{cfg.epochs}"
        epoch_started = time.time()
        lr_start = optimizer.param_groups[0]["lr"]
        train_loss = _run_epoch(
            model, _epoch_loader(train_ds, cfg, epoch), device, optimizer,
            label=f"{tag} train", log_every=log_every, max_grad_norm=cfg.max_grad_norm, scheduler=scheduler,
        )  # fmt: skip
        history["train_loss"].append(train_loss)
        val_loss = None
        if val_loader is not None:
            val_loss = _run_epoch(model, val_loader, device, label=f"{tag} val", log_every=log_every)
            history["val_loss"].append(val_loss)
        if epoch_checkpoint_dir is not None:
            Path(epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                Path(epoch_checkpoint_dir) / f"epoch_{epoch + 1:02d}.pt",
            )
        monitored = train_loss if val_loss is None else val_loss

        improved = monitored < best_loss
        if improved:
            best_epoch, best_loss, patience_left = epoch, monitored, cfg.early_stop_patience
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            patience_left -= 1
        vram = (
            f" peak_gpu_mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            if device.startswith("cuda")
            else ""
        )
        log.info(
            "%s done in %s: lr=%g->%g train_loss=%.4f val_loss=%s best=%.4f (epoch %d) %s patience_left=%d%s",
            tag, _hms(time.time() - epoch_started), lr_start, optimizer.param_groups[0]["lr"], train_loss, f"{val_loss:.4f}" if val_loss is not None else "n/a",
            best_loss, best_epoch + 1, "NEW BEST" if improved else "no improvement", patience_left, vram,
        )  # fmt: skip
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, train_loss, val_loss)
        if not improved and patience_left <= 0:
            log.info("early stopping after %d epochs", epoch + 1)
            break

    model.load_state_dict(best_state, strict=True)
    log.info(
        "finished in %s; keeping epoch %d (loss %.4f)", _hms(time.time() - started), best_epoch + 1, best_loss
    )
    return TrainResult(
        model=model.to("cpu").eval(), history=history, best_epoch=best_epoch, best_loss=float(best_loss)
    )
