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

import torch

from td_hot.evaluate import score_chips, score_image, sweep_confidence
from td_hot.export import build_retinanet
from td_hot.params import PostprocessParams
from td_hot.preprocess import read_rgb_chip

log = logging.getLogger("evaluate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True, type=Path, help="a state dict (.pt)")
    ap.add_argument("--chips", type=Path)
    ap.add_argument("--labels", type=Path, help="labels.json of the chips")
    ap.add_argument("--iou", type=float, nargs="+", default=[0.5, 0.4])
    ap.add_argument("--confidence", type=float, default=0.3, help="threshold for the P/R/F1 table")
    ap.add_argument(
        "--sweep", action="store_true", help="also sweep the confidence threshold (at the last --iou)"
    )
    ap.add_argument(
        "--nms",
        type=float,
        nargs="+",
        help="NMS thresholds to sweep with --sweep (default: the library's 0.05)",
    )
    ap.add_argument("--max-chips", type=int)
    ap.add_argument("--reference-image", type=Path)
    ap.add_argument("--reference-boxes", type=Path)
    ap.add_argument("--reference-scale", type=float, default=2.0)
    ap.add_argument("--reference-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", type=Path, help="write all results here")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    device = "cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda"
    model = build_retinanet(args.weights).to(device)
    out: dict = {"weights": str(args.weights), "device": device}

    if args.chips and args.labels:
        scores = score_chips(
            model, images_dir=args.chips, labels_json=args.labels, params={"confidence_threshold": args.confidence},
            iou_thresholds=tuple(args.iou), max_chips=args.max_chips,
        )  # fmt: skip
        out["chips"] = {f"iou{k}": v for k, v in scores.items()}
        if args.sweep:
            out["sweep"] = []
            for nms in args.nms or [PostprocessParams.nms_threshold]:
                rows = sweep_confidence(
                    model,
                    images_dir=args.chips,
                    labels_json=args.labels,
                    iou=args.iou[-1],
                    params={"nms_threshold": nms},
                    max_chips=args.max_chips,
                )
                out["sweep"] += [{"nms_threshold": nms, **row} for row in rows]
                for row in rows:
                    log.info(
                        "nms=%.2f confidence>%.2f: detections=%d precision=%.3f recall=%.3f f1=%.3f",
                        nms, row["threshold"], row["detections"], row["precision"], row["recall"], row["f1"],
                    )  # fmt: skip
            best = max(out["sweep"], key=lambda r: r["f1"])
            log.info(
                "best f1 %.3f at nms=%.2f confidence>%.2f",
                best["f1"],
                best["nms_threshold"],
                best["threshold"],
            )

    if args.reference_image and args.reference_boxes:
        rgb, _, _ = read_rgb_chip(args.reference_image)
        truth = [
            {"x1": float(r["xmin"]), "y1": float(r["ymin"]), "x2": float(r["xmax"]), "y2": float(r["ymax"])}
            for r in csv.DictReader(open(args.reference_boxes, encoding="utf-8"))
        ]
        rows = []
        for t in args.reference_thresholds:
            m = score_image(model, rgb, truth, scale=args.reference_scale, params={"confidence_threshold": t})
            rows.append({"threshold": t, **m})
            log.info(
                "reference image, confidence>%.2f: precision=%.3f recall=%.3f f1=%.3f",
                t,
                m["precision"],
                m["recall"],
                m["f1"],
            )
        out["reference"] = rows
        best = max(rows, key=lambda r: r["f1"])
        log.info("reference image: best f1 %.3f at confidence %.2f", best["f1"], best["threshold"])

    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
