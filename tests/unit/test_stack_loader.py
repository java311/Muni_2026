"""Pruebas del cargador de pilas (F1)."""

import numpy as np
import pytest
from PIL import Image

from muni.io.stack_loader import (
    list_images,
    load_folder,
    load_stack,
    natural_key,
    stack_hashes,
)


def test_natural_key_orders_numerically():
    names = ["imagen10.tif", "imagen2.tif", "imagen1.tif"]
    assert sorted(names, key=natural_key) == ["imagen1.tif", "imagen2.tif", "imagen10.tif"]


def _write_gray(path, value, size=(4, 3)):
    Image.new("L", size, color=value).save(path)


def test_load_stack_matches_dimensions(tmp_path):
    files = []
    for i in range(5):
        p = tmp_path / f"imagen{i}.png"
        _write_gray(p, value=i * 10)
        files.append(p)

    vol, resolved = load_stack(files, spacing_xy_um=0.5, spacing_z_um=2.0)
    assert vol.shape == (5, 3, 4)
    assert vol.spacing_xy_um == 0.5
    assert vol.plane(2)[0, 0] == pytest.approx(20.0)
    assert len(resolved) == 5


def test_load_stack_rejects_mixed_sizes(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_gray(a, 0, size=(4, 4))
    _write_gray(b, 0, size=(5, 4))
    with pytest.raises(ValueError, match="mismo tamaño"):
        load_stack([a, b])


def test_load_folder_natural_order_and_filter(tmp_path):
    _write_gray(tmp_path / "s1.png", 1)
    _write_gray(tmp_path / "s10.png", 10)
    _write_gray(tmp_path / "s2.png", 2)
    (tmp_path / "nota.txt").write_text("ignorar")

    vol, resolved = load_folder(tmp_path)
    assert [p.name for p in resolved] == ["s1.png", "s2.png", "s10.png"]
    assert vol.shape == (3, 3, 4)


def test_stack_hashes_deterministic(tmp_path):
    p = tmp_path / "h.png"
    _write_gray(p, 7)
    assert stack_hashes([p]) == stack_hashes([p])
