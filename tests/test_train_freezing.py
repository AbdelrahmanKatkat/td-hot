"""Which weights train: frozen encoder, the deepest stages only, or everything; BatchNorm never."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from torch.nn.modules.batchnorm import _BatchNorm  # noqa: E402

from td_hot.export import build_retinanet  # noqa: E402
from td_hot.train import _freeze, _optimizer_groups  # noqa: E402


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.fixture(scope="module")
def build():
    def _build(**kw):
        model = build_retinanet(None)
        for p in model.parameters():
            p.requires_grad_(True)
        _freeze(model, **kw)
        return model

    return _build


def test_more_unfrozen_stages_train_more_weights(build):
    frozen = trainable(build(freeze_encoder=True))
    layer4 = trainable(build(freeze_encoder=True, unfreeze_from="layer4"))
    layer3 = trainable(build(freeze_encoder=True, unfreeze_from="layer3"))
    everything = trainable(build(freeze_encoder=False))
    assert frozen < layer4 < layer3 < everything
    assert 8.0e6 < frozen < 9.5e6  # FPN + heads only
    assert layer4 - frozen > 14e6  # layer4 holds about 15 M weights


def test_the_right_stages_train(build):
    model = build(freeze_encoder=True, unfreeze_from="layer4")
    body = model.backbone.body
    assert body.layer4[0].conv1.weight.requires_grad
    assert not body.layer3[0].conv1.weight.requires_grad and not body.layer1[0].conv1.weight.requires_grad
    assert all(p.requires_grad for p in model.head.parameters())


def test_batchnorm_never_trains(build):
    for kwargs in ({"freeze_encoder": False}, {"freeze_encoder": True, "unfreeze_from": "layer1"}):
        model = build(**kwargs)
        norms = [m for m in model.modules() if isinstance(m, _BatchNorm)]
        assert norms and not any(p.requires_grad for m in norms for p in m.parameters())


def test_encoder_learning_rate_factor(build):
    model = build(freeze_encoder=True, unfreeze_from="layer4")
    groups = _optimizer_groups(model, 1e-3, 0.1)
    assert [g["lr"] for g in groups] == [pytest.approx(1e-3), pytest.approx(1e-4)]
    assert sum(p.numel() for g in groups for p in g["params"]) == trainable(model)
    assert len(_optimizer_groups(build(freeze_encoder=True), 1e-3, 0.1)) == 1  # nothing in the encoder trains
    assert isinstance(groups[0]["params"][0], torch.nn.Parameter)
