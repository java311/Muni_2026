"""Malla 3D a partir del esqueleto: tubos de radio variable + elipsoide del soma.

Equivale al "fitting" de SNT: la superficie se barre a lo largo de las ramas con
el radio estimado en cada nodo. También alberga la geometría de barrido que usa
el visor (``muni.view.skeleton_mesh``).
"""

from __future__ import annotations

import numpy as np

from muni.core.meshdata import MeshData
from muni.trace.base import TraceResult

SOMA_SLICES = 20
SOMA_STACKS = 12


def order_branch(idx: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Ordena una rama de nodo raíz a hoja."""
    members = set(idx.tolist())
    start = int(idx[0])
    for node in idx:
        p = int(parents[node])
        if p < 0 or p not in members:
            start = int(node)
            break
    child: dict[int, int] = {}
    for node in idx:
        p = int(parents[node])
        if p in members:
            child[p] = int(node)
    order = [start]
    cur = start
    while cur in child:
        cur = child[cur]
        order.append(cur)
    return np.asarray(order, dtype=np.int64)


def tube_geometry(
    points: np.ndarray, radii: np.ndarray, sides: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Barre un tubo de radio variable a lo largo de una polilínea.

    Devuelve ``(vértices, normales)`` como sopa de triángulos (``(T*3, 3)``).
    """
    m = len(points)
    if m < 2:
        return None
    segments = np.diff(points, axis=0)
    if np.all(np.linalg.norm(segments, axis=1) < 1e-9):
        return None

    tangents = np.empty_like(points)
    tangents[0] = segments[0]
    tangents[-1] = segments[-1]
    if m > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    lengths = np.linalg.norm(tangents, axis=1)
    lengths[lengths == 0] = 1.0
    tangents = tangents / lengths[:, None]

    # Marco de referencia transportado en paralelo para evitar torsión.
    normals = np.empty_like(points)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, tangents[0]))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    normals[0] = ref - np.dot(ref, tangents[0]) * tangents[0]
    normals[0] /= np.linalg.norm(normals[0])
    for i in range(1, m):
        axis = np.cross(tangents[i - 1], tangents[i])
        sin_a = float(np.linalg.norm(axis))
        cos_a = float(np.clip(np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0))
        if sin_a < 1e-9:
            normals[i] = normals[i - 1]
        else:
            axis = axis / sin_a
            normals[i] = (
                normals[i - 1] * cos_a
                + np.cross(axis, normals[i - 1]) * sin_a
                + axis * np.dot(axis, normals[i - 1]) * (1.0 - cos_a)
            )
        normals[i] -= np.dot(normals[i], tangents[i]) * tangents[i]
        length = float(np.linalg.norm(normals[i]))
        normals[i] = normals[i] / length if length > 1e-12 else normals[i - 1]

    binormals = np.cross(tangents, normals)
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    radial = np.cos(theta)[None, :, None] * normals[:, None, :] + np.sin(
        theta
    )[None, :, None] * binormals[:, None, :]
    ring = points[:, None, :] + radii[:, None, None] * radial

    j = np.arange(sides)
    jn = (j + 1) % sides
    tri1 = np.stack([ring[:-1, j], ring[:-1, jn], ring[1:, jn]], axis=2)
    tri2 = np.stack([ring[:-1, j], ring[1:, jn], ring[1:, j]], axis=2)
    n1 = np.stack([radial[:-1, j], radial[:-1, jn], radial[1:, jn]], axis=2)
    n2 = np.stack([radial[:-1, j], radial[1:, jn], radial[1:, j]], axis=2)

    verts = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0)
    norms = np.concatenate([n1.reshape(-1, 3), n2.reshape(-1, 3)], axis=0)
    lengths = np.linalg.norm(norms, axis=1)
    lengths[lengths == 0] = 1.0
    norms = norms / lengths[:, None]
    return verts, norms


def ellipsoid_geometry(
    center: np.ndarray,
    radii: np.ndarray,
    axes: np.ndarray | None,
    slices: int = SOMA_SLICES,
    stacks: int = SOMA_STACKS,
) -> tuple[np.ndarray, np.ndarray]:
    """Malla UV de un elipsoide orientado, como sopa de triángulos."""
    if axes is None:
        axes = np.eye(3)
    u = np.linspace(0.0, 2.0 * np.pi, slices, endpoint=False)
    v = np.linspace(0.0, np.pi, stacks + 1)
    su, sv = np.meshgrid(u, v, indexing="xy")

    local = np.stack(
        [np.sin(sv) * np.cos(su), np.sin(sv) * np.sin(su), np.cos(sv)], axis=-1
    )
    pos_local = local * radii[None, None, :]
    nrm_local = local / radii[None, None, :]
    length = np.linalg.norm(nrm_local, axis=-1, keepdims=True)
    length[length == 0] = 1.0
    nrm_local = nrm_local / length

    pos = pos_local @ axes.T + center
    nrm = nrm_local @ axes.T

    p00, p10 = pos[:-1], pos[1:]
    p01, p11 = np.roll(pos[:-1], -1, axis=1), np.roll(pos[1:], -1, axis=1)
    n00, n10 = nrm[:-1], nrm[1:]
    n01, n11 = np.roll(nrm[:-1], -1, axis=1), np.roll(nrm[1:], -1, axis=1)

    tri1 = np.stack([p00, p01, p11], axis=2)
    tri2 = np.stack([p00, p11, p10], axis=2)
    n1 = np.stack([n00, n01, n11], axis=2)
    n2 = np.stack([n00, n11, n10], axis=2)

    verts = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0)
    norms = np.concatenate([n1.reshape(-1, 3), n2.reshape(-1, 3)], axis=0)
    lengths = np.linalg.norm(norms, axis=1)
    lengths[lengths == 0] = 1.0
    norms = norms / lengths[:, None]
    return verts, norms


def tube_mesh_from_trace(
    trace: TraceResult,
    *,
    sides: int = 8,
    include_soma: bool = True,
) -> MeshData:
    """Construye una malla de tubos (y soma) a partir del esqueleto tipado.

    Nota: los tubos y el elipsoide son superficies cerradas que se intersecan,
    no una superficie fusionada (suficiente para GLB y medición).
    """
    empty = MeshData(
        vertices=np.empty((0, 3), dtype=np.float32),
        normals=np.empty((0, 3), dtype=np.float32),
        faces=np.empty((0, 3), dtype=np.int32),
        name="skeleton",
    )
    if trace.n_nodes == 0:
        return empty

    world = trace.world_coords()
    triangles: list[np.ndarray] = []
    vertex_normals: list[np.ndarray] = []

    for branch in np.unique(trace.branch_labels):
        idx = np.flatnonzero(trace.branch_labels == branch)
        if len(idx) < 2:
            continue
        order = order_branch(idx, trace.parents)
        tube = tube_geometry(
            world[order],
            np.clip(trace.radii[order], 1e-3, None),
            sides,
        )
        if tube is not None:
            verts, normals = tube
            triangles.append(verts.reshape(-1, 3, 3))
            vertex_normals.append(normals)

    if include_soma and trace.has_soma:
        verts, normals = ellipsoid_geometry(
            trace.soma_center_um, trace.soma_radii_um, trace.soma_axes
        )
        triangles.append(verts.reshape(-1, 3, 3))
        vertex_normals.append(normals)

    if not triangles:
        return empty

    return MeshData.from_triangle_soup(
        np.concatenate(triangles, axis=0),
        name="skeleton",
        vertex_normals=np.concatenate(vertex_normals, axis=0),
    )