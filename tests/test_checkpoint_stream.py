"""Weights can be given as an open binary stream (a download kept in memory) as well as a path."""

import io

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from td_hot.export import build_retinanet  # noqa: E402


def test_weights_load_from_a_binary_stream():
    source = build_retinanet()
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    loaded = build_retinanet(io.BytesIO(buffer.getvalue()))
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(source.state_dict().values(), loaded.state_dict().values())
    )
