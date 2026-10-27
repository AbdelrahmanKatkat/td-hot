"""Entry point to `td_hot.export`: write a DeepForest RetinaNet state dict as a level-2 ONNX file.

    python scripts/export_onnx.py weights/NEON.pt NEON.onnx

Needs the `train` extra (torch and onnx); the output runs with `scripts/predict.py` or `td_hot.predict_session`.
"""

import argparse
from pathlib import Path

from td_hot.export import build_retinanet, export_onnx_bytes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("weights", type=Path, help="a state dict (.pt)")
    parser.add_argument("out", type=Path, help="the ONNX file to write")
    return parser


def _export(weights: Path, out: Path) -> None:
    out.write_bytes(export_onnx_bytes(build_retinanet(weights)))


def main() -> None:
    args = _build_parser().parse_args()
    _export(args.weights, args.out)


if __name__ == "__main__":
    main()
