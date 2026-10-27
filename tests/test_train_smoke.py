"""One tiny fine-tune + eval pass on random weights (CPU, a few seconds): needs the `train` extra."""

import logging

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.integration

from td_hot.evaluate import score_dataset  # noqa: E402
from td_hot.params import parse_train_params  # noqa: E402
from td_hot.train import train  # noqa: E402


def test_train_and_evaluate_smoke(prepared_dataset, caplog):
    caplog.set_level(logging.INFO, logger="td_hot.train")
    out = prepared_dataset
    epochs_seen: list = []
    result = train(
        train_images_dir=out / "train" / "images",
        train_labels=out / "train" / "labels.json",
        val_images_dir=out / "val" / "images",
        val_labels=out / "val" / "labels.json",
        cfg=parse_train_params(
            {"epochs": 1, "device": "cpu", "batch_size": 1, "warmup_steps": 1, "epoch_fraction": 1.0}
        ),
        log_every=1,
        epoch_checkpoint_dir=out.parent / "epochs",
        on_epoch_end=lambda epoch, train, val: epochs_seen.append((epoch, train, val)),
    )
    assert len(result.history["train_loss"]) == 1 and result.best_loss > 0
    text = caplog.text
    assert (
        "start: device=cpu" in text
        and "epoch 1/1 train step" in text
        and "epoch 1/1 done" in text
        and "finished in" in text
    )
    assert not result.model.training
    assert [p.name for p in (out.parent / "epochs").iterdir()] == ["epoch_01.pt"]
    assert [e for e, _, _ in epochs_seen] == [1] and epochs_seen[0][2] == result.history["val_loss"][0]
    assert "freeze_encoder=True" in text and "trainable=" in text

    both = score_dataset(
        result.model,
        images_dir=out / "val" / "images",
        labels_json=out / "val" / "labels.json",
        iou_thresholds=(0.5, 0.4),
    )
    assert set(both) == {0.5, 0.4} and both[0.4]["tp"] >= both[0.5]["tp"]
    assert {"precision", "recall", "f1", "tp", "fp", "fn"} <= both[0.5].keys()


def test_train_from_the_original_chips_like_a_pipeline_step(prepared_dataset):
    """A pack step splits the original chips by name and passes the labels as a mapping: no copies, no shared files."""
    from td_hot.dataset import build_chip_labels, spatial_split

    root = prepared_dataset.parent
    chips = root / "chips"
    labels = build_chip_labels(chips, root / "labels")
    train_names, val_names = spatial_split(sorted(labels), 0.5, 42, block_size=4)
    seen = []
    result = train(
        train_images_dir=chips, train_labels=labels, train_names=train_names,
        val_images_dir=chips, val_labels=labels, val_names=val_names,
        cfg=parse_train_params({"epochs": 1, "device": "cpu", "batch_size": 1}),
        on_epoch_end=lambda e, tr, va: seen.append((e, tr, va)),
    )  # fmt: skip
    assert len(seen) == 1 and seen[0][2] is not None and result.best_loss > 0


def test_scoring_one_split_of_a_shared_chips_folder(prepared_dataset):
    """`names` + a labels mapping score only the validation chips of the original folder (what a pack step has)."""
    from td_hot.dataset import build_chip_labels
    from td_hot.evaluate import sweep_confidence
    from td_hot.export import build_retinanet

    root = prepared_dataset.parent
    labels = build_chip_labels(root / "chips", root / "labels")
    model = build_retinanet(None).eval()
    one = sorted(labels)[:1]
    scored = score_dataset(
        model, images_dir=root / "chips", labels_json=labels, names=one, params={"device": "cpu"}
    )
    assert scored[0.5]["tp"] + scored[0.5]["fn"] == len(labels[one[0]])
    rows = sweep_confidence(
        model, images_dir=root / "chips", labels_json=labels, names=one, thresholds=(0.05, 0.5)
    )
    assert [r["threshold"] for r in rows] == [0.05, 0.5]
