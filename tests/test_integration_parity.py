"""Opt-in parity check: numpy pipeline + ONNX vs torchvision RetinaNet with NEON.pt weights.

Run with `uv run pytest -m integration`. Requires weights/NEON.pt and the `train` extra.
"""

from pathlib import Path

import numpy as np
import pytest

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "NEON.pt"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def torch_model_and_session():
    import onnxruntime as ort

    from td_hot.export import build_retinanet, export_onnx_bytes

    assert WEIGHTS.is_file(), f"missing {WEIGHTS}"
    model = build_retinanet(WEIGHTS)
    session = ort.InferenceSession(export_onnx_bytes(model), providers=["CPUExecutionProvider"])
    return model, session


def _synthetic_rgb(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))[:h, :w]
    return img


@pytest.mark.parametrize("hw", [(400, 400), (256, 256), (300, 420)])
def test_onnx_raw_outputs_match_torch(torch_model_and_session, hw):
    import torch

    from td_hot.preprocess import preprocess_window

    model, session = torch_model_and_session
    rgb = _synthetic_rgb(*hw)
    batch, _, _ = preprocess_window(rgb)

    with torch.no_grad():
        feats = list(model.backbone(torch.from_numpy(batch)).values())
        ref = model.head(feats)
    cls_onnx, box_onnx = session.run(None, {"images": batch})
    np.testing.assert_allclose(cls_onnx, ref["cls_logits"].numpy(), atol=2e-3, rtol=1e-3)
    np.testing.assert_allclose(box_onnx, ref["bbox_regression"].numpy(), atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("hw", [(400, 400), (256, 256), (300, 420)])
def test_preprocess_matches_torchvision_transform(torch_model_and_session, hw):
    import torch

    from td_hot.preprocess import preprocess_window

    model, _ = torch_model_and_session
    rgb = _synthetic_rgb(*hw)
    batch, new_w, new_h = preprocess_window(rgb)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    images, _ = model.transform([tensor])
    ref = images.tensors.numpy()
    assert batch.shape == ref.shape
    assert images.image_sizes[0] == (new_h, new_w)
    assert np.abs(batch - ref).max() < 5e-2  # PIL vs torch bilinear differ slightly on noisy images


@pytest.mark.parametrize("hw", [(400, 400), (256, 256), (300, 420)])
def test_numpy_decode_matches_torchvision_detections(torch_model_and_session, hw):
    import torch

    from td_hot.decode import decode_retinanet_output
    from td_hot.params import PostprocessParams
    from td_hot.preprocess import preprocess_window

    model, session = torch_model_and_session
    rgb = _synthetic_rgb(*hw)
    batch, new_w, new_h = preprocess_window(rgb)
    cfg = PostprocessParams(confidence_threshold=0.02, nms_threshold=0.05)

    # Same normalised tensor through the torchvision head + postprocess_detections.
    model.score_thresh, model.nms_thresh = cfg.confidence_threshold, cfg.nms_threshold
    with torch.no_grad():
        t = torch.from_numpy(batch)
        feats = list(model.backbone(t).values())
        head = model.head(feats)
        split = [f.shape[-2] * f.shape[-1] * 9 for f in feats]
        head_out = {k: list(v.split(split, dim=1)) for k, v in head.items()}
        from torchvision.models.detection.image_list import ImageList

        anchors = model.anchor_generator(ImageList(t, [(new_h, new_w)]), feats)
        per_level = [list(a.split(split)) for a in anchors]  # as RetinaNet.forward does
        ref = model.postprocess_detections(head_out, per_level, [(new_h, new_w)])[0]

    cls_onnx, box_onnx = session.run(None, {"images": batch})
    ours = decode_retinanet_output(cls_onnx, box_onnx, batch.shape[2:], (new_h, new_w), cfg)

    assert len(ours) == len(ref["boxes"])
    if ours:
        np.testing.assert_allclose([d["confidence"] for d in ours], ref["scores"].numpy(), atol=2e-3)
        np.testing.assert_allclose(
            [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in ours], ref["boxes"].numpy(), atol=0.5
        )
