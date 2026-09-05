"""Pruebas del núcleo de datos (F1): volumen, malla y GLTF."""

import math

import numpy as np
import pytest

from muni.core.gltf_io import load_model, read_glb, save_model, write_glb
from muni.core.meshdata import MeshData
from muni.core.meta import ModelMeta
from muni.core.volume import Volume3D


# ---------------------------------------------------------------- Volume3D
def test_volume_shape_and_spacing():
    vol = Volume3D(np.zeros((4, 8, 6), dtype=np.float32), spacing_xy_um=0.5, spacing_z_um=2.0)
    assert vol.shape == (4, 8, 6)
    assert (vol.dimz, vol.dimy, vol.dimx) == (4, 8, 6)
    assert vol.spacing_xy_um == 0.5
    assert vol.plane(1).shape == (8, 6)
    assert vol.physical_size() == (4 * 2.0, 8 * 0.5, 6 * 0.5)


def test_volume_rejects_2d_and_bad_spacing():
    with pytest.raises(ValueError):
        Volume3D(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        Volume3D(np.zeros((2, 2, 2)), spacing_z_um=0)


# ---------------------------------------------------------------- MeshData
def test_mesh_from_triangle_soup_welds_and_computes_normals():
    # Tetraedro regular-ish con 4 triángulos, compartiendo vértices.
    tris = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        ],
        dtype=np.float32,
    )
    mesh = MeshData.from_triangle_soup(tris)
    assert mesh.vertex_count == 4
    assert mesh.face_count == 4
    assert mesh.validate() == []
    # Las normales deben ser unitarias.
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)


def test_mesh_validate_detects_degenerate():
    mesh = MeshData(
        vertices=np.zeros((3, 3), dtype=np.float32),
        normals=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 0, 0]], dtype=np.int32),
    )
    assert any("degenerados" in e for e in mesh.validate())


# ---------------------------------------------------------------- GLTF
def test_glb_roundtrip(tmp_path):
    tris = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        ],
        dtype=np.float32,
    )
    mesh = MeshData.from_triangle_soup(tris, name="tetra")
    path = tmp_path / "tetra.glb"
    write_glb(mesh, path)
    assert path.exists()

    back = read_glb(path)
    assert back.vertex_count == mesh.vertex_count
    assert back.face_count == mesh.face_count
    assert back.name == "tetra"
    # Comparación geométrica (los índices pueden permutarse tras el weld).
    assert np.allclose(back.vertices, mesh.vertices)
    assert np.allclose(back.normals, mesh.normals, atol=1e-5)


def test_save_and_load_model_with_sidecar(tmp_path):
    tris = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    mesh = MeshData.from_triangle_soup(tris, name="solo")
    meta = ModelMeta(dimx=10, dimy=10, dimz=10, spacing_xy_um=0.4, spacing_z_um=1.1)
    glb = tmp_path / "solo.glb"
    save_model(glb, mesh, meta)

    assert (tmp_path / "solo.muni.json").exists()
    back_mesh, back_meta = load_model(glb)
    assert back_mesh.face_count == 1
    assert back_meta is not None
    assert back_meta.dimx == 10
    assert back_meta.spacing_xy_um == pytest.approx(0.4)


# ---------------------------------------------------------------- utilidades
def test_mesh_bounds():
    mesh = MeshData(
        vertices=np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32),
        normals=np.ones((2, 3), dtype=np.float32),
        faces=np.array([[0, 1, 0]], dtype=np.int32),
    )
    lo, hi = mesh.bounds()
    assert np.allclose(lo, [0, 0, 0])
    assert np.allclose(hi, [2, 4, 6])
