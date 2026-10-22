"""Torch model and ONNX export of the DeepForest RetinaNet (needs the `train` extra).

The exported graph is the "level 2" contract:

    input  `image`  (3, H, W) float32 in [0, 1], RGB, already scaled by `preprocess.py`
    output `boxes`  (N, 4) XYXY in input pixels, `scores` (N,), `labels` (N,), best score first

Inside the graph: ImageNet normalisation, padding to 32, backbone + FPN + head, anchors, sigmoid,
top-1000 per level, box decode and clipping, i.e. torchvision's own code traced, not re-implemented.
Left out on purpose: the confidence filter, NMS and the per-image cap (`decode.decode_detections`),
so they stay tunable. The score floor inside the graph is 0, which is equivalent to DeepForest's 0.1
floor followed by the same filter outside, because top-k keeps the highest scores either way.
"""

import logging
from pathlib import Path

from .params import IMAGE_MEAN, IMAGE_STD, TOPK_CANDIDATES, PostprocessParams

log = logging.getLogger(__name__)

OPSET = 17
INPUT_NAME = "image"
OUTPUT_NAMES = ("boxes", "scores", "labels")


def build_retinanet(weights_path: str | Path | None = None):
    """torchvision RetinaNet-ResNet50-FPN with 1 class; optionally load a DeepForest state dict.

    The checkpoint stores weights only. Normalisation, anchors and thresholds are set here, so every
    generation of the model (base, fine-tuned, re-fine-tuned) is rebuilt with the same constants.
    Resizing happens outside the model (`preprocess.py`), so torchvision's resize step is disabled:
    training, evaluation and export all see the same pixels as serving.
    """
    import torch
    from torchvision.models.detection import retinanet_resnet50_fpn

    defaults = PostprocessParams()
    model = retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=1,
        score_thresh=defaults.confidence_threshold,
        nms_thresh=defaults.nms_threshold,
        topk_candidates=TOPK_CANDIDATES,
        image_mean=list(IMAGE_MEAN),
        image_std=list(IMAGE_STD),
    )
    model.transform.resize = lambda image, target=None: (image, target)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.eval()


def level2_network(model):
    """Wrap `model` (switched to eval mode) as `image -> (boxes, scores, labels)` candidates before NMS.

    Reuses `RetinaNet.postprocess_detections` with the score floor at 0, no cap, and NMS replaced by
    a plain sort by score, so the numbers come from torchvision itself.
    """
    import torch
    from torchvision.models.detection import retinanet

    def keep_all_by_score(boxes, scores, idxs, iou_threshold):
        return torch.argsort(scores, descending=True)

    class Level2(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image):
            m = self.inner
            saved = (m.score_thresh, m.detections_per_img, retinanet.box_ops.batched_nms)
            m.score_thresh, m.detections_per_img = 0.0, 10**6
            retinanet.box_ops.batched_nms = keep_all_by_score
            try:
                out = m([image])[0]
            finally:
                m.score_thresh, m.detections_per_img, retinanet.box_ops.batched_nms = saved
            return out["boxes"], out["scores"], out["labels"]

    # eval() recurses into `model`: torch.onnx.export restores the wrapper's mode afterwards, so a
    # wrapper left in train mode would put the caller's model back into train mode.
    return Level2(model).eval()


def export_onnx_bytes(model, *, opset: int = OPSET) -> bytes:
    """Export `level2_network(model)` with dynamic height and width; validate and return bytes."""
    import tempfile

    import onnx
    import torch

    net = level2_network(model)
    dummy = torch.rand(3, 800, 800)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "td_hot.onnx"
        torch.onnx.export(
            net,
            (dummy,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={
                INPUT_NAME: {1: "height", 2: "width"},
                OUTPUT_NAMES[0]: {0: "candidates"},
                OUTPUT_NAMES[1]: {0: "candidates"},
                OUTPUT_NAMES[2]: {0: "candidates"},
            },
            opset_version=opset,
            dynamo=False,
        )
        proto = onnx.load(path)
        onnx.save_model(proto, path, save_as_external_data=False)
        onnx.checker.check_model(str(path))
        data = path.read_bytes()
    log.info(
        "exported ONNX (opset %d, level 2, inputs %s -> outputs %s): %.1f MB",
        opset,
        INPUT_NAME,
        OUTPUT_NAMES,
        len(data) / 1e6,
    )
    return data


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Export a DeepForest RetinaNet state dict to ONNX (level 2)")
    ap.add_argument("weights")
    ap.add_argument("out")
    args = ap.parse_args()
    Path(args.out).write_bytes(export_onnx_bytes(build_retinanet(args.weights)))


if __name__ == "__main__":
    main()
