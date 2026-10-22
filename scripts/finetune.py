"""Entry point to `td_hot.train`: fine-tune a checkpoint and write `<name>.pt` and `<name>.onnx`.

    python scripts/finetune.py --train-dir data/guru_boxes/train --val-dir data/guru_boxes/val \
        --weights weights/NEON.pt --out data/runs/full --name NEON_OAM --epochs 3 --batch-size 4

A dataset folder holds `chips/` (or `images/`) and `labels.json`. Every training option that is left out keeps the
default of `td_hot.params.TrainParams`. Needs the `train` extra; with a local CUDA torch use `--no-sync` / the venv
python, not a plain `uv run`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from td_hot.export import export_onnx_bytes
from td_hot.params import parse_preprocess_params, parse_train_params
from td_hot.train import train_retinanet


def dataset_dirs(root: Path) -> tuple[Path, Path]:
    images = root / "chips" if (root / "chips").is_dir() else root / "images"
    return images, root / "labels.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--val-dir", type=Path, help="enables validation loss and best-epoch selection")
    ap.add_argument("--weights", required=True, type=Path, help="the checkpoint to start from (a state dict)")
    ap.add_argument("--out", required=True, type=Path, help="output folder; must be empty or new")
    ap.add_argument("--name", default="NEON_OAM", help="output file stem: <name>.pt and <name>.onnx")
    ap.add_argument("--model-gsd-m", type=float, help="metres per model pixel (must match serving)")
    ap.add_argument("--log-every", type=int, default=0, help="0 = about 50 step lines per epoch")
    for flag, kind in (("epochs", int), ("batch-size", int), ("lr", float), ("weight-decay", float),
                       ("max-grad-norm", float), ("scheduler", str), ("sample-fraction", float),
                       ("patience", int), ("device", str), ("seed", int), ("warmup-steps", int), ("epoch-fraction", float)):  # fmt: skip
        ap.add_argument(f"--{flag}", type=kind)
    ap.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--unfreeze-from",
        help="with a frozen encoder, still train ResNet stages from this one: layer4, layer3, ...",
    )
    ap.add_argument(
        "--encoder-lr-factor", type=float, help="learning-rate multiplier for the encoder weights that train"
    )
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; choose a new --out")
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.out / "train.log", encoding="utf-8"),
        ],
    )

    hyper = {
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm, "scheduler": args.scheduler,
        "sample_fraction": args.sample_fraction, "early_stop_patience": args.patience, "device": args.device,
        "seed": args.seed, "freeze_encoder": args.freeze_encoder,
        "warmup_steps": args.warmup_steps, "epoch_fraction": args.epoch_fraction,
        "unfreeze_from": args.unfreeze_from, "encoder_lr_factor": args.encoder_lr_factor,
    }  # fmt: skip
    cfg = parse_train_params({k: v for k, v in hyper.items() if v is not None})  # None -> TrainParams default
    pre = parse_preprocess_params({"model_gsd_m": args.model_gsd_m})
    train_images, train_labels = dataset_dirs(args.train_dir)
    val_images, val_labels = dataset_dirs(args.val_dir) if args.val_dir else (None, None)

    result = train_retinanet(
        train_images_dir=train_images, train_labels=train_labels,
        val_images_dir=val_images, val_labels=val_labels,
        cfg=cfg, pre=pre, checkpoint_path=args.weights, log_every=args.log_every,
        epoch_checkpoint_dir=args.out / "epochs",
    )  # fmt: skip

    pt, onnx = args.out / f"{args.name}.pt", args.out / f"{args.name}.onnx"
    torch.save(result.model.state_dict(), pt)
    onnx.write_bytes(export_onnx_bytes(result.model))
    (args.out / "history.json").write_text(
        json.dumps(
            {**result.history, "best_epoch": result.best_epoch + 1, "best_loss": result.best_loss}, indent=2
        ),
        encoding="utf-8",
    )
    logging.getLogger("finetune").info(
        "wrote %s (%.1f MB) and %s (%.1f MB)", pt, pt.stat().st_size / 1e6, onnx, onnx.stat().st_size / 1e6
    )


if __name__ == "__main__":
    main()
