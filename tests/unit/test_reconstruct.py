"""Pruebas de extracción de superficie (F3)."""

import numpy as np
import pytest

from muni.core.meshdata import MeshData
from muni.postprocess.smooth import laplacian_smooth
from muni.reconstruct.dual_contouring import DualContouring
from muni.reconstruct.marching_cubes import MarchingCubes


def _sphere_field(shape=(24, 24, 24), center=(12, 12, 12), radius=8.0, scale=1.0):
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float32)
    d = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    return (1.0 / (1.0 + np.exp((d - radius) / scale))).astype(np.float32)


def test_marching_cubes_sphere():
    field = _sphere_field()
    mc = MarchingCubes(inside_high=True)
    result = mc.extract(field, isovalue=0.5, spacing=(1.0, 1.0, 1.0))
    mesh = result.mesh
    assert not mesh.is_empty
    assert mesh.validate() == []

    center = np.array([12.0, 12.0, 12.0], dtype=np.float32)
    radial = mesh.vertices - center
    dist = np.linalg.norm(radial, axis=1)
    # Todos los vértices deben caer cerca del radio (tolerancia ~ diagonal de celda).
    assert np.allclose(dist, 8.0, atol=1.8)

    # Las normales deben apuntar hacia fuera (dot(n, radial) > 0).
    dot = np.einsum("ij,ij->i", mesh.normals, radial)
    assert (dot > 0).mean() > 0.95


def test_dual_contouring_sphere():
    field = _sphere_field()
    dc = DualContouring(inside_high=True)
    result = dc.extract(field, isovalue=0.5, spacing=(1.0, 1.0, 1.0))
    mesh = result.mesh
    assert not mesh.is_empty

    center = np.array([12.0, 12.0, 12.0], dtype=np.float32)
    radial = mesh.vertices - center
    dist = np.linalg.norm(radial, axis=1)
    assert np.allclose(dist, 8.0, atol=1.8)

    # Normales hacia fuera.
    dot = np.einsum("ij,ij->i", mesh.normals, radial)
    assert (dot > 0).mean() > 0.95


def test_marching_cubes_empty_field():
    field = np.zeros((6, 6, 6), dtype=np.float32)  # sin superficie
    mc = MarchingCubes(inside_high=True)
    result = mc.extract(field, isovalue=0.5)
    assert result.mesh.is_empty


def test_marching_cubes_rejects_2d():
    mc = MarchingCubes()
    with pytest.raises(ValueError):
        mc.extract(np.zeros((6, 6), dtype=np.float32))


def test_laplacian_smooth_moves_vertices_toward_neighbors():
    # Pirámide simple: un vértice desplazado que debería acercarse a la base.
    verts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 5.0]],
        dtype=np.float32,
    )
    norms = np.ones_like(verts)
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 1, 2]], dtype=np.int32)
    mesh = MeshData(vertices=verts, normals=norms, faces=faces)

    out = laplacian_smooth(mesh, iterations=4, factor=0.5)
    assert np.all(np.isfinite(out.vertices))
    # El vértice 3 (punta) baja hacia el plano z=0.
    assert out.vertices[3, 2] < verts[3, 2]
