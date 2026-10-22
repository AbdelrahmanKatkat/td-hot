"""Entry point to `td_hot.predict_session`: run an exported ONNX on a folder of chips and write GeoJSON (CPU).

    python scripts/predict.py --onnx data/runs/full/NEON_OAM.onnx --chips data/guru_boxes/test/chips \
        --out predictions.geojson --params '{"confidence_threshold": 0.1}'

Chips are named `OAM-<x>-<y>-<z>.<ext>`. Needs `onnxruntime` (the `infer` extra); no torch.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import onnxruntime as ort

from td_hot import predict_session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--chips", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--params", default="{}", help='JSON, for example {"confidence_threshold": 0.1}')
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    collection = predict_session(session, args.chips, json.loads(args.params))
    args.out.write_text(json.dumps(collection), encoding="utf-8")
    logging.getLogger("predict").info("wrote %d features to %s", len(collection["features"]), args.out)


if __name__ == "__main__":
    main()
