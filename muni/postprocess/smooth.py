"""Suavizado de mallas implementado desde cero (sin Open3D/trimesh).

Opera sobre la topología existente (no remalla): solo desplaza vértices.
"""

from __future__ import annotations

import numpy as np

from muni.core.meshdata import MeshData


def _vertex_neighbors(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(suma_de_vecinos, conteo)`` por vértice.

    Usa las aristas de los triángulos (sin duplicar direcciones).
    """
    f = mesh.faces
    e01 = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.sort(e01, axis=1)

    n = mesh.vertex_count
    neighbor_sum = np.zeros((n, 3), dtype=np.float32)
    count = np.zeros(n, dtype=np.int32)
    v = mesh.vertices

    np.add.at(neighbor_sum, e[:, 0], v[e[:, 1]])
    np.add.at(neighbor_sum, e[:, 1], v[e[:, 0]])
    np.add.at(count, e[:, 0], 1)
    np.add.at(count, e[:, 1], 1)
    return neighbor_sum, count


def laplacian_smooth(
    mesh: MeshData, iterations: int = 5, factor: float = 0.5, inplace: bool = False
) -> MeshData:
    """Suavizado Laplaciano (operador paraguas): ``p <- p + λ·(media_vecinos - p)``.

    Devuelve una copia (o modifica ``mesh`` si ``inplace=True``).
    """
    verts = mesh.vertices if inplace else mesh.vertices.copy()
    neighbor_sum, count = _vertex_neighbors(mesh)

    valid = count > 0
    for _ in range(iterations):
        centroid = np.zeros_like(verts)
        np.divide(neighbor_sum, count[:, None], out=centroid, where=valid[:, None])
        verts[valid] = verts[valid] + factor * (centroid[valid] - verts[valid])

    if inplace:
        mesh.vertices = verts
        return mesh
    return MeshData(
        vertices=verts,
        normals=mesh.normals.copy(),
        faces=mesh.faces.copy(),
        name=mesh.name,
    )


def taubin_smooth(
    mesh: MeshData,
    iterations: int = 10,
    lam: float = 0.5,
    mu: float = -0.53,
    inplace: bool = False,
) -> MeshData:
    """Suavizado de Taubin (alterna contracción λ y expansión μ) para evitar encogimiento."""
    out = mesh if inplace else MeshData(
        vertices=mesh.vertices.copy(),
        normals=mesh.normals.copy(),
        faces=mesh.faces.copy(),
        name=mesh.name,
    )
    for _ in range(iterations):
        laplacian_smooth(out, iterations=1, factor=lam, inplace=True)
        laplacian_smooth(out, iterations=1, factor=mu, inplace=True)
    return out
