"""The `scripts/` entry points run end to end as real processes (random weights, CPU, a few seconds each)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.integration

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(script: str, *args: str) -> str:
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)], capture_output=True, text=True
    )
    assert done.returncode == 0, f"{script} failed:\n{done.stdout[-1500:]}\n{done.stderr[-1500:]}"
    return done.stdout + done.stderr


def test_finetune_evaluate_predict_scripts(prepared_dataset, write_chip, tmp_path):
    import torch

    from td_hot.export import build_retinanet

    init = tmp_path / "init.pt"
    torch.save(build_retinanet(None).state_dict(), init)  # random weights stand in for NEON.pt
    out = tmp_path / "run"

    log = run(
        "finetune.py", "--train-dir", prepared_dataset / "train", "--val-dir", prepared_dataset / "val",
        "--weights", init, "--out", out, "--name", "NEON_OAM", "--epochs", "1", "--batch-size", "1", "--device", "cpu",
    )  # fmt: skip
    assert "start: device=cpu" in log and "epoch 1/1 done" in log and "wrote" in log
    assert (
        (out / "NEON_OAM.pt").is_file()
        and (out / "NEON_OAM.onnx").is_file()
        and (out / "train.log").is_file()
    )
    assert json.loads((out / "history.json").read_text())["best_epoch"] == 1

    chip = write_chip(name="OAM-524288-524288-20.tif")  # a reference image with one labelled box
    boxes = tmp_path / "ref.csv"
    with open(boxes, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["xmin", "ymin", "xmax", "ymax"])
        w.writerow([10, 10, 40, 40])
    report = tmp_path / "report.json"
    run(
        "evaluate.py", "--weights", out / "NEON_OAM.pt", "--chips", prepared_dataset / "val" / "images",
        "--labels", prepared_dataset / "val" / "labels.json", "--iou", "0.5", "0.4", "--sweep", "--nms", "0.05", "0.3", "--device", "cpu",
        "--reference-image", chip, "--reference-boxes", boxes, "--reference-scale", "1.0",
        "--reference-thresholds", "0.05", "0.2", "--json", report,
    )  # fmt: skip
    results = json.loads(report.read_text())
    assert (
        set(results["chips"]) == {"iou0.5", "iou0.4"}
        and len(results["sweep"]) == 12
        and {r["nms_threshold"] for r in results["sweep"]} == {0.05, 0.3}
    )
    assert [r["threshold"] for r in results["reference"]] == [0.05, 0.2]

    predictions = tmp_path / "pred.geojson"
    run(
        "predict.py", "--onnx", out / "NEON_OAM.onnx", "--chips", prepared_dataset / "val" / "images",
        "--out", predictions, "--params", '{"confidence_threshold": 0.0}',
    )  # fmt: skip
    collection = json.loads(predictions.read_text())
    assert collection["type"] == "FeatureCollection"
