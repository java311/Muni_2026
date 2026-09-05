"""Pruebas de preprocesado (F2)."""

import numpy as np
import pytest

from muni.preprocess.denoise import anisotropic_diffusion, denoise_volume, normalize
from muni.preprocess.enhance import contrast_stretch, histogram_equalize
from muni.core.volume import Volume3D


def test_normalize_range():
    img = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
    out = normalize(img, (0.0, 1.0))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_anisotropic_diffusion_reduces_noise_keeps_edge():
    rng = np.random.default_rng(0)
    # Bloque con un borde: izquierda 10, derecha 200, más ruido gaussiano.
    img = np.where(np.arange(64)[None, :] < 32, 10.0, 200.0)
    noisy = img + rng.normal(0, 8.0, size=(64, 64)).astype(np.float32)
    out = anisotropic_diffusion(noisy, iterations=8, kappa=20.0, gamma=0.2)
    assert out.shape == noisy.shape
    assert np.all(np.isfinite(out))
    # La varianza en una región plana debe bajar.
    left_before = noisy[:, 8:24].var()
    left_after = out[:, 8:24].var()
    assert left_after < left_before
    # El contraste del borde debe conservarse en gran medida.
    edge_before = abs(noisy[:, 31].mean() - noisy[:, 32].mean())
    edge_after = abs(out[:, 31].mean() - out[:, 32].mean())
    assert edge_after > 0.5 * edge_before


def test_histogram_equalize_increases_contrast():
    # Imagen concentrada en un rango estrecho.
    img = (np.linspace(0, 1, 256) ** 3 * 255).astype(np.float32)[None, :].repeat(8, axis=0)
    out = histogram_equalize(img)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(255.0)
    assert out.std() > img.std()


def test_contrast_stretch_expands_range():
    img = np.array([[100.0, 110.0], [120.0, 130.0]], dtype=np.float32)
    out = contrast_stretch(img)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(255.0)


def test_denoise_volume_returns_volume():
    vol = Volume3D(np.ones((3, 8, 8), dtype=np.float32) * 128)
    out = denoise_volume(vol, iterations=2)
    assert isinstance(out, Volume3D)
    assert out.shape == vol.shape
