"""Tracing dendrítico usando scikit-image (skeletonize) + esqueleto tipado.

El resultado es un **bosque** de esqueletos (uno por componente conexo de la
máscara) con nodos tipados (soma, axón, dendrita). La raíz se sitúa en el soma
detectado, no en un punto final arbitrario.

La aritmética de grafo vive en :mod:`muni.trace.graph` (compartida con la
edición interactiva del esqueleto).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as _label
from skimage.morphology import skeletonize

from muni.trace.base import SWC_AXON, SWC_DENDRITE, SWC_SOMA, Tracer, TraceResult
from muni.trace.graph import (
    branch_labels,
    branch_start,
    build_adjacency,
    build_forest,
    children_of,
    compact,
    prune_spurs,
    to_world,
    total_length,
)
from muni.trace.soma import clamp_soma, detect_soma


def _classify(
    parents: np.ndarray,
    radii: np.ndarray,
    branch_labels: np.ndarray,
    node_comp: np.ndarray,
    soma,
    center_grid: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Asigna tipos SWC por rama: soma, axón (fino) y dendritas.

    El soma es el nodo más cercano a la esfera detectada. El axón es la rama
    claramente más fina que la referencia (mediana de las ramas primarias) y se
    propaga por sus ramas hijas mientras sigan finas; el resto son dendritas.

    ponytail: heurística; la clasificación definitiva es manual (paso de edición).
    """
    n = len(parents)
    types = np.full(n, SWC_DENDRITE, dtype=np.int64)
    if soma is None:
        return types

    world = to_world(center_grid, spacing)
    dist = np.linalg.norm(world - soma.center_um, axis=1)
    root = int(np.argmin(dist))
    if dist[root] > 4.0 * soma.radius_um:
        return types

    types[root] = SWC_SOMA
    soma_comp = node_comp[root]

    branch_radius: dict[int, float] = {}
    members: dict[int, np.ndarray] = {}
    parent_branch: dict[int, int] = {}
    start_node: dict[int, int] = {}
    for branch in np.unique(branch_labels):
        idx = np.flatnonzero(branch_labels == branch)
        if node_comp[idx[0]] != soma_comp:
            continue
        key = int(branch)
        start = branch_start(idx, parents)
        # Mediana, no media: la parte proximal (dentro del soma) es gruesa y no
        # debe dominar el grosor característico de la rama.
        branch_radius[key] = float(np.median(radii[idx]))
        members[key] = idx
        start_node[key] = start
        p = int(parents[start])
        parent_branch[key] = int(branch_labels[p]) if p >= 0 else -1

    primary = [
        b
        for b, start in start_node.items()
        if int(parents[start]) == root and len(members[b]) >= 2
    ]
    if not primary:
        return types
    reference = float(np.median([branch_radius[b] for b in primary]))
    if reference <= 0.0:
        return types

    # Axón: la rama primaria claramente más fina (candidata) y más larga.
    thinnest = min(primary, key=lambda b: branch_radius[b])
    best_branch: int | None = None
    if branch_radius[thinnest] <= 0.7 * reference:
        best_branch = thinnest

    if best_branch is not None:
        branch_children: dict[int, list[int]] = {}
        for b, pb in parent_branch.items():
            branch_children.setdefault(pb, []).append(b)
        threshold = 1.3 * branch_radius[best_branch]
        stack = [best_branch]
        marked: set[int] = set()
        while stack:
            b = stack.pop()
            if b in marked:
                continue
            marked.add(b)
            for child in branch_children.get(b, []):
                if branch_radius.get(child, 0.0) <= threshold:
                    stack.append(child)
        for b in marked:
            types[members[b]] = SWC_AXON

    types[root] = SWC_SOMA
    return types


def _empty_result(spacing: tuple[float, float, float]) -> TraceResult:
    return TraceResult(
        coords=np.empty((0, 3)),
        radii=np.empty(0),
        parents=np.empty(0, dtype=np.int64),
        branch_labels=np.empty(0, dtype=np.int64),
        total_length_um=0.0,
        spacing=spacing,
        method="skimage",
    )


class SkimageTracer(Tracer):
    """Tracing con skeletonize (Lee94): forest + soma + tipos SWC."""

    name = "skimage"

    def trace(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        *,
        min_spur_um: float = 2.0,
    ) -> TraceResult:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3:
            raise ValueError(f"mask debe ser 3D, se recibió forma {mask.shape}.")
        if mask.sum() == 0:
            raise ValueError("mask está vacía (todos False).")
        spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

        soma = detect_soma(mask, spacing)

        skeleton = skeletonize(mask)
        edt = distance_transform_edt(mask, sampling=spacing)

        coords_grid, adj_list, degree = build_adjacency(skeleton)
        n = len(coords_grid)
        if n == 0:
            return _empty_result(spacing)

        radii = np.array([edt[z, y, x] for z, y, x in coords_grid], dtype=np.float64)

        # Acotar el soma al grosor de las neuritas: el pico de la EDT puede caer
        # en un cruce de neuritas gruesas y quedar sobredimensionado.
        soma = clamp_soma(soma, float(np.median(radii)))

        # Componente de la máscara a la que pertenece cada voxel del esqueleto.
        comp, _ = _label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        node_comp = comp[coords_grid[:, 0], coords_grid[:, 1], coords_grid[:, 2]]

        parents = build_forest(coords_grid, adj_list, degree, radii, node_comp, soma, spacing)

        # Poda de espolones del esqueleto (ruido), nunca del tronco.
        keep = prune_spurs(coords_grid, parents, spacing, min_spur_um)
        coords_grid, parents, radii, node_comp = compact(
            coords_grid, parents, radii, node_comp, keep
        )
        n = len(coords_grid)
        if n == 0:
            return _empty_result(spacing)

        children = children_of(parents)
        labels = branch_labels(parents, children)
        length = total_length(coords_grid, parents, spacing)

        types = _classify(parents, radii, labels, node_comp, soma, coords_grid, spacing)

        return TraceResult(
            coords=coords_grid.astype(np.float64),
            radii=radii,
            parents=parents,
            branch_labels=labels,
            total_length_um=length,
            spacing=spacing,
            method=self.name,
            types=types,
            soma_center_um=soma.center_um if soma is not None else None,
            soma_radii_um=soma.radii_um if soma is not None else None,
            soma_axes=soma.axes if soma is not None else None,
        )