"""`build_chip_labels` and `DetectionDataset`: training straight from the original chips, without prepared copies."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from td_hot.dataset import build_chip_labels  # noqa: E402
from td_hot.params import PreprocessParams  # noqa: E402
from td_hot.detection_dataset import DetectionDataset  # noqa: E402


@pytest.fixture
def originals(prepared_dataset):
    root = prepared_dataset.parent
    return root / "chips", root / "labels"


def test_labels_for_chips_matches_the_prepared_labels(prepared_dataset, originals):
    chips, labels = originals
    mapping = build_chip_labels(chips, labels)
    prepared = {}
    for split in ("train", "val"):
        prepared.update(json.loads((prepared_dataset / split / "labels.json").read_text()))
    assert mapping == prepared
    assert (
        sum(len(v) for v in mapping.values()) == 2
    )  # the one box sits in both chips (same ground footprint)


def test_dataset_takes_a_mapping_and_a_named_subset(originals):
    chips, labels = originals
    mapping = build_chip_labels(chips, labels)
    names = sorted(mapping)[:1]
    ds = DetectionDataset(chips, mapping, PreprocessParams(), names=names)
    assert [p.name for p in ds.paths] == names
    image, target = ds[0]
    assert image.shape[0] == 3 and target["boxes"].shape == (len(mapping[names[0]]), 4)
    with pytest.raises(ValueError, match="not found"):
        DetectionDataset(chips, mapping, PreprocessParams(), names=["OAM-1-1-20.tif"])


def test_path_and_mapping_labels_give_the_same_target(prepared_dataset):
    images = prepared_dataset / "train" / "images"
    labels_path = prepared_dataset / "train" / "labels.json"
    from_path = DetectionDataset(images, labels_path, PreprocessParams())
    from_map = DetectionDataset(images, json.loads(labels_path.read_text()), PreprocessParams())
    np.testing.assert_array_equal(from_path[0][1]["boxes"].numpy(), from_map[0][1]["boxes"].numpy())


def test_flip_augmentation_mirrors_image_and_boxes(prepared_dataset):
    images, labels = prepared_dataset / "train" / "images", prepared_dataset / "train" / "labels.json"
    plain = DetectionDataset(images, labels, PreprocessParams())
    flipped = DetectionDataset(images, labels, PreprocessParams(), augment=True)

    class AlwaysFlip:
        def random(self):
            return 0.0

    flipped.rng = AlwaysFlip()
    (img0, t0), (img1, t1) = plain[0], flipped[0]
    width = img0.shape[2]
    np.testing.assert_allclose(img1.numpy(), img0.numpy()[:, :, ::-1])
    b0, b1 = t0["boxes"].numpy(), t1["boxes"].numpy()
    np.testing.assert_allclose(b1[:, [0, 2]], width - b0[:, [2, 0]], atol=1e-4)
    np.testing.assert_allclose(b1[:, [1, 3]], b0[:, [1, 3]])
