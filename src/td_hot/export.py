"""ONNX export of the DeepForest NEON checkpoint (needs the `train` extra)."""

from pathlib import Path

OPSET = 17
INPUT_NAME = "images"
OUTPUT_NAMES = ("cls_logits", "bbox_regression")


def build_retinanet(weights_path: str | Path | None = None):
    """torchvision RetinaNet-ResNet50-FPN with 1 class; optionally load a DeepForest state dict."""
    import torch
    from torchvision.models.detection import retinanet_resnet50_fpn

    model = retinanet_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=1, score_thresh=0.1, nms_thresh=0.05
    )
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.eval()


def _raw_network(model):
    """Backbone + FPN + head on an already-normalised, padded tensor -> raw head outputs."""
    import torch

    class RawRetinaNet(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.backbone = inner.backbone
            self.head = inner.head

        def forward(self, images):
            feats = list(self.backbone(images).values())
            out = self.head(feats)
            return out["cls_logits"], out["bbox_regression"]

    return RawRetinaNet(model).eval()


def export_onnx_bytes(model, *, opset: int = OPSET) -> bytes:
    """Export the raw network with dynamic H/W and anchor axes; validate and return bytes.

    Resize, normalisation, anchors, box decoding and NMS stay in numpy (preprocess/decode).
    """
    import tempfile

    import onnx
    import torch

    raw = _raw_network(model)
    dummy = torch.zeros(1, 3, 800, 800)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "td_hot.onnx"
        torch.onnx.export(
            raw,
            dummy,
            str(path),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={
                INPUT_NAME: {2: "height", 3: "width"},
                OUTPUT_NAMES[0]: {1: "anchors"},
                OUTPUT_NAMES[1]: {1: "anchors"},
            },
            opset_version=opset,
            dynamo=False,
        )
        proto = onnx.load(path)
        onnx.save_model(proto, path, save_as_external_data=False)
        onnx.checker.check_model(str(path))
        return path.read_bytes()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Export DeepForest NEON.pt to ONNX")
    ap.add_argument("weights")
    ap.add_argument("out")
    args = ap.parse_args()
    Path(args.out).write_bytes(export_onnx_bytes(build_retinanet(args.weights)))


if __name__ == "__main__":
    main()
