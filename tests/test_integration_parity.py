"""Opt-in parity check: level-2 ONNX + numpy decode vs the torchvision RetinaNet with NEON.pt weights.

Run with `uv run pytest -m integration`. Requires weights/NEON.pt and the `train` extra.
"""

from pathlib import Path

import numpy as np
import pytest

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "NEON.pt"

pytestmark = pytest.mark.integration

SIZES = [(764, 764), (800, 800), (600, 840)]  # model-input sizes: a z20 chip, 256 px x 3.125, non-square


@pytest.fixture(scope="module")
def model_and_onnx():
    import onnxruntime as ort

    from td_hot.export import build_retinanet, export_onnx_bytes

    assert WEIGHTS.is_file(), f"missing {WEIGHTS}"
    model = build_retinanet(WEIGHTS)
    onnx_bytes = export_onnx_bytes(model)
    return model, onnx_bytes, ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])


def _image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Blocky noise in [0, 1], CHW float32: enough structure for the head to fire on something."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    rgb = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))[:h, :w]
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)


def test_graph_has_anchors_and_decode_but_no_nms(model_and_onnx):
    import onnx

    _, onnx_bytes, session = model_and_onnx
    ops = {n.op_type for n in onnx.load_from_string(onnx_bytes).graph.node}
    assert "NonMaxSuppression" not in ops and "TopK" in ops  # level 2: ranked candidates, no NMS
    assert [i.name for i in session.get_inputs()] == ["image"]
    assert [o.name for o in session.get_outputs()] == ["boxes", "scores", "labels"]
    assert session.get_inputs()[0].shape[0] == 3 and isinstance(session.get_inputs()[0].shape[1], str)


@pytest.mark.parametrize("hw", SIZES)
def test_onnx_candidates_match_torch(model_and_onnx, hw):
    import torch

    from td_hot.export import level2_network

    model, _, session = model_and_onnx
    image = _image(*hw)
    boxes, scores, labels = session.run(None, {"image": image})
    with torch.no_grad():
        rb, rs, rl = level2_network(model)(torch.from_numpy(image))
    assert boxes.shape == tuple(rb.shape) and boxes.shape[0] > 0
    np.testing.assert_allclose(scores, rs.numpy(), atol=2e-3)
    # Thousands of candidates have near-tied scores, so compare them as a set, not by position.
    ref_boxes = rb.numpy()
    worst = max(np.abs(ref_boxes - box).max(axis=1).min() for box in boxes)
    assert worst < 0.5
    assert (labels == rl.numpy()).all()
    assert (np.diff(scores) <= 1e-6).all()  # best score first


@pytest.mark.parametrize("hw", SIZES)
@pytest.mark.parametrize("threshold", [0.02, 0.1])
def test_numpy_decode_matches_torchvision_detections(model_and_onnx, hw, threshold):
    import torch

    from td_hot.decode import decode_detections
    from td_hot.params import PostprocessParams

    model, _, session = model_and_onnx
    image = _image(*hw)
    cfg = PostprocessParams(confidence_threshold=threshold, nms_threshold=0.05)

    model.score_thresh, model.nms_thresh = cfg.confidence_threshold, cfg.nms_threshold
    with torch.no_grad():
        ref = model([torch.from_numpy(image)])[0]
    model.score_thresh, model.nms_thresh = 0.1, 0.05

    ours = decode_detections(*session.run(None, {"image": image}), cfg)
    assert len(ours) == len(ref["boxes"])
    if ours:
        np.testing.assert_allclose([d["confidence"] for d in ours], ref["scores"].numpy(), atol=2e-3)
        np.testing.assert_allclose(
            [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in ours], ref["boxes"].numpy(), atol=0.5
        )
