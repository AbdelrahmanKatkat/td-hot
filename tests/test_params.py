import pytest

from td_hot.params import (
    parse_inference_params,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)


@pytest.mark.parametrize("empty", [None, {}, {"confidence_threshold": None}])
def test_defaults_for_empty_params(empty):
    p = parse_inference_params(empty)
    assert (p.patch_size, p.patch_overlap) == (400, 0.05)
    assert (p.confidence_threshold, p.nms_threshold, p.iou_threshold) == (0.1, 0.05, 0.15)


def test_overrides_and_string_values():
    p = parse_inference_params({"patch_size": "300", "patch_overlap": 0.25, "iou_threshold": "0.3"})
    assert (p.patch_size, p.patch_overlap, p.iou_threshold) == (300, 0.25, 0.3)


def test_alias_and_precedence():
    assert parse_postprocess_params({"score_thresh": 0.3}).confidence_threshold == 0.3
    both = {"confidence_threshold": 0.5, "score_thresh": 0.3}
    assert parse_postprocess_params(both).confidence_threshold == 0.5  # canonical wins
    assert parse_postprocess_params({"nms_thresh": 0.2}).nms_threshold == 0.2


@pytest.mark.parametrize(
    "bad",
    [{"confidence_threshold": 1.5}, {"iou_threshold": -0.1}, {"confidence_threshold": True}],
)
def test_invalid_postprocess_values(bad):
    with pytest.raises(ValueError):
        parse_postprocess_params(bad)


@pytest.mark.parametrize("bad", [{"patch_size": 8}, {"patch_overlap": 1.0}, {"patch_overlap": -0.1}])
def test_invalid_preprocess_values(bad):
    with pytest.raises(ValueError):
        parse_preprocess_params(bad)


def test_split_params():
    s = parse_split_params(None)
    assert (s.val_ratio, s.split_seed, s.block_size) == (0.2, 42, 4)
    assert parse_split_params({"val_ratio": 0.3, "split_seed": 7}).split_seed == 7
    with pytest.raises(ValueError):
        parse_split_params({"val_ratio": 0.0})


def test_train_params_defaults_and_validation():
    cfg = parse_train_params({})
    assert cfg.epochs >= 1 and cfg.batch_size >= 1
    with pytest.raises(ValueError):
        parse_train_params({"epochs": 0})


def test_train_params_overrides_alias_and_bounds():
    cfg = parse_train_params({"lr": "0.001", "freeze_backbone": "true", "epochs": 3, "seed": None})
    assert cfg.learning_rate == 0.001 and cfg.freeze_backbone is True and cfg.epochs == 3 and cfg.seed == 42
    for bad in (
        {"learning_rate": 0},
        {"sample_fraction": 1.5},
        {"early_stop_patience": 0},
        {"freeze_backbone": "maybe"},
    ):
        with pytest.raises(ValueError):
            parse_train_params(bad)
