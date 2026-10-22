import pytest

from td_hot.params import (
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)


@pytest.mark.parametrize("empty", [None, {}, {"confidence_threshold": None}])
def test_defaults_for_empty_params(empty):
    p = parse_postprocess_params(empty)
    assert (p.confidence_threshold, p.nms_threshold) == (0.1, 0.05)
    assert parse_preprocess_params(empty).model_gsd_m == 0.05


def test_alias_and_precedence():
    assert parse_postprocess_params({"score_thresh": 0.3}).confidence_threshold == 0.3
    both = {"confidence_threshold": 0.5, "score_thresh": 0.3}
    assert parse_postprocess_params(both).confidence_threshold == 0.5  # canonical wins
    assert parse_postprocess_params({"nms_thresh": "0.2"}).nms_threshold == 0.2


@pytest.mark.parametrize(
    "bad",
    [{"confidence_threshold": 1.5}, {"nms_threshold": -0.1}, {"confidence_threshold": True}],
)
def test_invalid_postprocess_values(bad):
    with pytest.raises(ValueError):
        parse_postprocess_params(bad)


def test_preprocess_params_override_and_bounds():
    assert parse_preprocess_params({"model_gsd_m": "0.1"}).model_gsd_m == 0.1
    for bad in ({"model_gsd_m": 0}, {"model_gsd_m": 5}, {"model_gsd_m": True}):
        with pytest.raises(ValueError):
            parse_preprocess_params(bad)


def test_split_params():
    s = parse_split_params(None)
    assert (s.val_ratio, s.split_seed, s.block_size) == (0.2, 42, 4)
    assert parse_split_params({"val_ratio": 0.3, "split_seed": 7}).split_seed == 7
    with pytest.raises(ValueError):
        parse_split_params({"val_ratio": 0.0})


def test_train_params_defaults_are_the_pack_recipe():
    cfg = parse_train_params({})
    assert (cfg.learning_rate, cfg.weight_decay, cfg.max_grad_norm, cfg.scheduler) == (
        1e-3,
        1e-4,
        1.0,
        "cosine",
    )
    assert cfg.freeze_encoder is True and cfg.epochs >= 1 and cfg.batch_size >= 1
    with pytest.raises(ValueError):
        parse_train_params({"epochs": 0})


def test_train_params_overrides_alias_and_bounds():
    cfg = parse_train_params(
        {"lr": "0.01", "freeze_encoder": "false", "epochs": 3, "seed": None, "scheduler": "None"}
    )
    assert cfg.learning_rate == 0.01 and cfg.freeze_encoder is False and cfg.epochs == 3 and cfg.seed == 42
    assert cfg.scheduler == "none"
    assert (
        parse_train_params({"freeze_backbone": False}).freeze_encoder is False
    )  # earlier name still accepted
    for bad in (
        {"learning_rate": 0},
        {"sample_fraction": 1.5},
        {"early_stop_patience": 0},
        {"freeze_encoder": "maybe"},
        {"scheduler": "step"},
        {"max_grad_norm": -1},
    ):
        with pytest.raises(ValueError):
            parse_train_params(bad)


def test_partial_unfreeze_params():
    cfg = parse_train_params({"unfreeze_from": "Layer4", "encoder_lr_factor": "0.1"})
    assert (cfg.freeze_encoder, cfg.unfreeze_from, cfg.encoder_lr_factor) == (True, "layer4", 0.1)
    assert parse_train_params({}).unfreeze_from == "none" and parse_train_params({}).encoder_lr_factor == 1.0
    for bad in ({"unfreeze_from": "layer9"}, {"encoder_lr_factor": 0}, {"encoder_lr_factor": 2}):
        with pytest.raises(ValueError):
            parse_train_params(bad)
    with pytest.raises(ValueError, match="needs freeze_encoder=True"):
        parse_train_params({"freeze_encoder": False, "unfreeze_from": "layer4"})
