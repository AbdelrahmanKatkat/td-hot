"""The CPU inference image installs only the core dependencies: importing the package must not need torch."""

import subprocess
import sys

HEAVY = ("torch", "torchvision", "onnx", "datasets", "pandas", "pyarrow", "onnxruntime")


def test_import_td_hot_loads_no_training_or_dataset_framework():
    code = (
        "import sys, td_hot;"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules];"
        "assert not bad, bad;"
        "assert callable(td_hot.predict_session) and callable(td_hot.decode_detections)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]


def test_serving_and_evaluation_modules_import_without_torch():
    code = (
        "import sys, td_hot.serve, td_hot.evaluate, td_hot.dataset, td_hot.tiles, td_hot.decode;"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules];"
        "assert not bad, bad"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
