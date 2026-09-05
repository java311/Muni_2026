"""Pruebas del motor U-Net opcional (F7)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from muni.core.volume import Volume3D  # noqa: E402
from muni.segment.unet import UnetSegmenter, _build_unet_module  # noqa: E402


def _synthetic_volume():
    z, y, x = np.mgrid[0:10, 0:28, 0:28].astype(np.float32)
    d = np.sqrt((z - 5) ** 2 + (y - 14) ** 2 + (x - 14) ** 2)
    field = (1.0 / (1.0 + np.exp((d - 7.0) / 1.0))).astype(np.float32)
    return Volume3D(field * 255.0, name="synth")


def _train_tiny(tmp_path) -> str:
    vol = _synthetic_volume()
    data = vol.data / 255.0
    labels = (data > 0.5).astype(np.float32)[:, None]

    model = _build_unet_module(3, 4)  # base mínima para el test
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    half = 1
    for step in range(3):
        model.train()
        for z in range(vol.dimz):
            stack = np.zeros((1, 3, vol.dimy, vol.dimx), dtype=np.float32)
            for k in range(3):
                zk = min(max(z + k - half, 0), vol.dimz - 1)
                stack[0, k] = data[zk]
            pred = model(torch.as_tensor(stack))
            loss = criterion(pred, torch.as_tensor(labels[z : z + 1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    path = tmp_path / "unet_tiny.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def test_unet_requires_weights(tmp_path):
    seg = UnetSegmenter(model_path=tmp_path / "no_existe.pt")
    with pytest.raises(RuntimeError, match="pesos"):
        seg.segment(_synthetic_volume())


def test_unet_inference(tmp_path):
    weights = _train_tiny(tmp_path)
    seg = UnetSegmenter(model_path=weights, base_features=4)

    result = seg.segment(_synthetic_volume())
    assert result.probability.shape == (10, 28, 28)
    assert result.probability.min() >= 0.0 and result.probability.max() <= 1.0
    assert result.mask.dtype == bool
    assert result.method == "unet"
    # El soma esférico central debe capturarse ligeramente.
    assert result.mask[:, 10:18, 10:18].any()


def test_unet_batched_inference_matches(tmp_path):
    weights = _train_tiny(tmp_path)
    seg = UnetSegmenter(model_path=weights, base_features=4)
    a = seg.segment(_synthetic_volume())
    b = seg.batched_segment(_synthetic_volume(), batch_size=4)
    assert np.allclose(a.probability, b.probability, atol=1e-6)