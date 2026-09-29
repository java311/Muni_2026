"""Geometría 3D del esqueleto para el viewport: tubos tipados + elipsoide del soma.

Devuelve una sopa de triángulos con normales por vértice, agrupada por color
(tipo SWC), para dibujarla con el mismo shader de la malla. La geometría de
barrido (tubos/elipsoide) vive en :mod:`muni.reconstruct.tube_mesh`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from muni.reconstruct.tube_mesh import ellipsoid_geometry, order_branch, tube_geometry
from muni.trace.base import SWC_AXON, SWC_DENDRITE, SWC_SOMA, SWC_SPINE, TraceResult

TYPE_COLORS: dict[int, tuple[float, float, float]] = {
    SWC_SOMA: (0.86, 0.22, 0.22),  # rojo
    SWC_AXON: (0.25, 0.78, 0.30),  # verde
    SWC_DENDRITE: (0.36, 0.56, 0.95),  # azul
    SWC_SPINE: (0.95, 0.75, 0.20),  # naranja
}
DEFAULT_COLOR = (0.80, 0.80, 0.80)
HIGHLIGHT_COLOR = (1.0, 0.5, 0.0)  # rama seleccionada / a borrar (naranja)
CONTEXT_COLOR = (1.0, 0.85, 0.15)  # padres (contexto, no se borran) (amarillo)


def build_skeleton_geometry(
    result: TraceResult,
    *,
    tube_sides: int = 8,
    soma_slices: int = 20,
    soma_stacks: int = 12,
    highlight_branches: set[int] | None = None,
    context_branches: set[int] | None = None,
    highlight_soma: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, tuple[float, float, float]]]]:
    """Construye ``(posiciones, normales, grupos)`` del esqueleto de ``result``.

    ``grupos`` es una lista de ``(offset, count, color)`` en número de vértices.
    Las ramas en ``highlight_branches`` (selección) se pintan con
    ``HIGHLIGHT_COLOR``; las de ``context_branches`` (padres, contexto) con
    ``CONTEXT_COLOR``; el soma con ``highlight_soma`` usa el color de selección.
    """
    highlight = {int(b) for b in highlight_branches} if highlight_branches else set()
    context = {int(b) for b in context_branches} if context_branches else set()
    world = result.world_coords()
    groups: dict[tuple[float, float, float], list[tuple[np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )

    for branch in np.unique(result.branch_labels):
        idx = np.flatnonzero(result.branch_labels == branch)
        if len(idx) < 2:
            continue
        order = order_branch(idx, result.parents)
        tube = tube_geometry(
            world[order],
            np.clip(result.radii[order], 1e-3, None),
            tube_sides,
        )
        if tube is not None:
            if int(branch) in highlight:
                color = HIGHLIGHT_COLOR
            elif int(branch) in context:
                color = CONTEXT_COLOR
            else:
                color = TYPE_COLORS.get(_branch_type(result.types[order]), DEFAULT_COLOR)
            groups[color].append(tube)

    if result.has_soma:
        verts, normals = ellipsoid_geometry(
            result.soma_center_um,
            result.soma_radii_um,
            result.soma_axes,
            soma_slices,
            soma_stacks,
        )
        color = HIGHLIGHT_COLOR if highlight_soma else TYPE_COLORS[SWC_SOMA]
        groups[color].append((verts, normals))

    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    ranges: list[tuple[int, int, tuple[float, float, float]]] = []
    offset = 0
    for color, chunks in groups.items():
        verts = np.concatenate([c[0] for c in chunks], axis=0)
        nrm = np.concatenate([c[1] for c in chunks], axis=0)
        positions.append(verts)
        normals.append(nrm)
        ranges.append((offset, len(verts), color))
        offset += len(verts)

    if not positions:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), []
    return (
        np.concatenate(positions, axis=0).astype(np.float32),
        np.concatenate(normals, axis=0).astype(np.float32),
        ranges,
    )


def _branch_type(types: np.ndarray) -> int:
    """Tipo dominante de la rama (el soma no cuenta: se dibuja como elipsoide)."""
    values = types[types != SWC_SOMA]
    if values.size == 0:
        return SWC_SOMA
    return int(np.bincount(values).argmax())