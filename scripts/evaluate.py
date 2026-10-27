"""Entry point to `td_hot.evaluate`: score a checkpoint on labelled chips and, optionally, on a reference image.

    python scripts/evaluate.py --weights data/runs/full/NEON_OAM.pt --chips data/guru_boxes/val/chips \
        --labels data/guru_boxes/val/labels.json --iou 0.5 0.4 --confidence 0.3 --sweep --nms 0.05 0.2 0.5 \
        --reference-image data/reference/OSBS_029.tif --reference-boxes data/reference/OSBS_029.csv --reference-scale 2.0

The reference image is for the forgetting check: score the base checkpoint and the fine-tuned one on the imagery the
base model came from (boxes as a CSV with columns xmin,ymin,xmax,ymax; the scale is that image's own, not the zoom
rule). Needs the `train` extra.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from td_hot.evaluate import score_chip, score_dataset, sweep_confidence
from td_hot.export import build_retinanet
from td_hot.params import PostprocessParams
from td_hot.preprocess import read_chip

log = logging.getLogger("evaluate")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", required=True, type=Path, help="a state dict (.pt)")
    parser.add_argument("--chips", type=Path)
    parser.add_argument("--labels", type=Path, help="labels.json of the chips")
    parser.add_argument("--iou", type=float, nargs="+", default=[0.5, 0.4])
    parser.add_argument("--confidence", type=float, default=0.3, help="threshold for the P/R/F1 table")
    parser.add_argument(
        "--sweep", action="store_true", help="also sweep the confidence threshold (at the last --iou)"
    )
    parser.add_argument(
        "--nms",
        type=float,
        nargs="+",
        help="NMS thresholds to sweep with --sweep (default: the library's 0.05)",
    )
    parser.add_argument("--max-chips", type=int)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--reference-boxes", type=Path)
    parser.add_argument("--reference-scale", type=float, default=2.0)
    parser.add_argument("--reference-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", type=Path, help="write all results here")
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _device(requested: str) -> str:
    return "cpu" if requested == "cpu" or not torch.cuda.is_available() else "cuda"


def _read_reference_truth(boxes_csv: Path) -> list[dict[str, float]]:
    with open(boxes_csv, encoding="utf-8") as handle:
        return [
            {
                "x1": float(row["xmin"]),
                "y1": float(row["ymin"]),
                "x2": float(row["xmax"]),
                "y2": float(row["ymax"]),
            }
            for row in csv.DictReader(handle)
        ]


def _sweep_nms(model: Any, args: argparse.Namespace, results: dict[str, Any]) -> None:
    """Confidence sweep at each NMS threshold (at the last --iou); logs every row and the best one."""
    results["sweep"] = []
    for nms in args.nms or [PostprocessParams.nms_threshold]:
        rows = sweep_confidence(
            model,
            images_dir=args.chips,
            labels_json=args.labels,
            iou=args.iou[-1],
            params={"nms_threshold": nms},
            max_chips=args.max_chips,
        )
        results["sweep"] += [{"nms_threshold": nms, **row} for row in rows]
        for row in rows:
            log.info(
                "nms=%.2f confidence>%.2f: detections=%d precision=%.3f recall=%.3f f1=%.3f",
                nms, row["threshold"], row["detections"], row["precision"], row["recall"], row["f1"],
            )  # fmt: skip
    best = max(results["sweep"], key=lambda row: row["f1"])
    log.info(
        "best f1 %.3f at nms=%.2f confidence>%.2f",
        best["f1"],
        best["nms_threshold"],
        best["threshold"],
    )


def _score_chips(model: Any, args: argparse.Namespace, results: dict[str, Any]) -> None:
    scores = score_dataset(
        model, images_dir=args.chips, labels_json=args.labels, params={"confidence_threshold": args.confidence},
        iou_thresholds=tuple(args.iou), max_chips=args.max_chips,
    )  # fmt: skip
    results["chips"] = {f"iou{iou}": metrics for iou, metrics in scores.items()}
    if args.sweep:
        _sweep_nms(model, args, results)


def _score_reference(model: Any, args: argparse.Namespace, results: dict[str, Any]) -> None:
    """Forgetting check: the same metrics on the reference image, at each confidence threshold."""
    rgb, _, _ = read_chip(args.reference_image)
    truth = _read_reference_truth(args.reference_boxes)
    reference_rows = []
    for threshold in args.reference_thresholds:
        metrics = score_chip(
            model, rgb, truth, scale=args.reference_scale, params={"confidence_threshold": threshold}
        )
        reference_rows.append({"threshold": threshold, **metrics})
        log.info(
            "reference image, confidence>%.2f: precision=%.3f recall=%.3f f1=%.3f",
            threshold,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
    results["reference"] = reference_rows
    best = max(reference_rows, key=lambda row: row["f1"])
    log.info("reference image: best f1 %.3f at confidence %.2f", best["f1"], best["threshold"])


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    model = build_retinanet(args.weights).to(device)
    results: dict[str, Any] = {"weights": str(args.weights), "device": device}
    if args.chips and args.labels:
        _score_chips(model, args, results)
    if args.reference_image and args.reference_boxes:
        _score_reference(model, args, results)
    return results


def main() -> None:
    args = _build_parser().parse_args()
    _configure_logging()
    results = _evaluate(args)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
