"""Selección de nodos del esqueleto en el visor 3D.

La selección se hace proyectando los nodos a pantalla y buscando el más cercano
al clic dentro de una tolerancia en píxeles. Es lógica pura (sin Qt/OpenGL), por
lo que se puede probar con una :class:`~muni.view.camera.OrbitCamera` cualquiera.
"""

from __future__ import annotations

import numpy as np

from muni.trace.base import TraceResult
from muni.view.camera import OrbitCamera


def nearest_node(
    result: TraceResult | None,
    camera: OrbitCamera,
    x: float,
    y: float,
    width: int,
    height: int,
    tolerance_px: float = 12.0,
) -> int | None:
    """Índice del nodo del esqueleto más cercano al clic, o ``None``.

    Solo se consideran nodos dentro del volumen de la cámara (NDC z en [-1, 1]).
    """
    if result is None or result.n_nodes == 0 or width <= 0 or height <= 0:
        return None

    world = result.world_coords()
    ndc = camera.project_points(world)
    valid = np.isfinite(ndc).all(axis=1) & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    if not valid.any():
        return None

    sx = (ndc[:, 0] + 1.0) * 0.5 * width
    sy = (1.0 - ndc[:, 1]) * 0.5 * height
    dist = np.full(len(world), np.inf)
    dist[valid] = np.hypot(sx[valid] - x, sy[valid] - y)

    idx = int(np.argmin(dist))
    if not np.isfinite(dist[idx]) or dist[idx] > tolerance_px:
        return None
    return idx