"""Edición del esqueleto: podar, borrar/añadir ramas, tipar y refinar radios.

Todas las operaciones son **inmutables**: devuelven un :class:`TraceResult`
nuevo (los arrays por nodo se compactan y las ramas se reetiquetan), de modo
que la UI puede apilar versiones para deshacer.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array

from muni.trace.base import SWC_DENDRITE, SWC_SOMA, TraceResult
from muni.trace.graph import (
    branch_labels,
    branch_start,
    children_of,
    compact,
    prune_spurs,
    to_world,
    total_length,
)


def _rebuild(
    trace: TraceResult,
    coords: np.ndarray,
    parents: np.ndarray,
    radii: np.ndarray,
    types: np.ndarray,
) -> TraceResult:
    """Reconstruye el resultado reetiquetando ramas y recalculando la longitud."""
    children = children_of(parents)
    labels = branch_labels(parents, children)
    return TraceResult(
        coords=coords,
        radii=radii,
        parents=parents,
        branch_labels=labels,
        total_length_um=total_length(coords, parents, trace.spacing),
        spacing=trace.spacing,
        method=trace.method,
        types=types,
        soma_center_um=trace.soma_center_um,
        soma_radii_um=trace.soma_radii_um,
        soma_axes=trace.soma_axes,
    )


def _subtree(parents: np.ndarray, root: int) -> np.ndarray:
    """Índices de ``root`` y todos sus descendientes."""
    children = children_of(parents)
    removed: set[int] = set()
    stack = [int(root)]
    while stack:
        node = stack.pop()
        if node in removed:
            continue
        removed.add(node)
        stack.extend(children[node])
    return np.array(sorted(removed), dtype=np.int64)


def prune_trace(trace: TraceResult, min_spur_um: float) -> TraceResult:
    """Elimina espolones terminales más cortos que ``min_spur_um``."""
    if trace.n_nodes == 0 or min_spur_um <= 0.0:
        return trace
    keep = prune_spurs(trace.coords, trace.parents, trace.spacing, min_spur_um)
    coords, parents, radii, _ = compact(
        trace.coords, trace.parents, trace.radii, trace.branch_labels, keep
    )
    types = trace.types[keep]
    return _rebuild(trace, coords, parents, radii, types)


def delete_branch(trace: TraceResult, branch_id: int) -> TraceResult:
    """Elimina la rama ``branch_id`` y todo su subárbol.

    Se borra desde el primer nodo de la rama (aquel cuyo padre queda fuera de
    ella), de modo que no quedan fragmentos colgando.
    """
    idx = np.flatnonzero(trace.branch_labels == branch_id)
    if idx.size == 0:
        raise ValueError(f"La rama {branch_id} no existe.")
    start = branch_start(idx, trace.parents)
    removed = _subtree(trace.parents, start)
    keep = np.ones(trace.n_nodes, dtype=bool)
    keep[removed] = False
    coords, parents, radii, _ = compact(
        trace.coords, trace.parents, trace.radii, trace.branch_labels, keep
    )
    types = trace.types[keep]
    return _rebuild(trace, coords, parents, radii, types)


def set_branch_type(trace: TraceResult, branch_id: int, swc_type: int) -> TraceResult:
    """Asigna el tipo SWC ``swc_type`` a todos los nodos de la rama."""
    idx = np.flatnonzero(trace.branch_labels == branch_id)
    if idx.size == 0:
        raise ValueError(f"La rama {branch_id} no existe.")
    types = trace.types.copy()
    types[idx] = int(swc_type)
    return dataclasses.replace(trace, types=types)


def _ray_radius(
    data: np.ndarray,
    origin_world: np.ndarray,
    directions: np.ndarray,
    spacing: tuple[float, float, float],
    isolevel: float,
    step: float,
    limit: float,
) -> list[float]:
    """Radio por rayo: distancia a la primera muestra fuera de la neurita."""
    shape = data.shape
    crossings: list[float] = []
    for direction in directions:
        s = step
        crossing = None
        while s <= limit:
            world = origin_world + s * direction
            iz = round(world[2] / spacing[0])
            iy = round(world[1] / spacing[1])
            ix = round(world[0] / spacing[2])
            if not (0 <= iz < shape[0] and 0 <= iy < shape[1] and 0 <= ix < shape[2]):
                break
            if float(data[iz, iy, ix]) > isolevel:
                crossing = s
                break
            s += step
        if crossing is not None:
            crossings.append(crossing)
    return crossings


def refit_radii(
    trace: TraceResult,
    data: np.ndarray,
    spacing: tuple[float, float, float],
    isolevel: float,
    *,
    max_probe_factor: float = 3.0,
) -> TraceResult:
    """Ajusta los radios a la sección transversal real de la imagen (SNT "fit").

    Para cada nodo se lanzan 4 rayos perpendiculares a la tangente y se toma la
    mediana de las distancias hasta el isonivel (neurona oscura: fuera cuando
    ``> isolevel``). Los nodos de soma conservan su radio.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"data debe ser 3D (Z, Y, X), se recibió {data.shape}.")
    if trace.n_nodes == 0:
        return trace

    spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    step = max(0.5 * min(max(s, 1e-6) for s in spacing), 1e-3)

    coords = trace.coords
    children = children_of(trace.parents)
    world = to_world(coords, spacing)
    radii = trace.radii.copy()

    for i in range(trace.n_nodes):
        if trace.types[i] == SWC_SOMA:
            continue
        p = int(trace.parents[i])
        ch = children[i]
        if p >= 0 and ch:
            tangent = world[ch[0]] - world[p]
        elif p >= 0:
            tangent = world[i] - world[p]
        elif ch:
            tangent = world[ch[0]] - world[i]
        else:
            continue
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-9:
            continue
        tangent = tangent / norm

        ref = np.array([0.0, 0.0, 1.0]) if abs(float(tangent[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(tangent, ref)
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-9:
            continue
        u = u / u_norm
        v = np.cross(tangent, u)
        directions = np.array([u, -u, v, -v])

        limit = max_probe_factor * max(float(radii[i]), step)
        crossings = _ray_radius(data, world[i], directions, spacing, isolevel, step, limit)
        if crossings:
            radii[i] = float(np.median(crossings))

    # Un paso de mediana sobre la cadena para suavizar nodos de bifurcación.
    smoothed = radii.copy()
    for i in range(trace.n_nodes):
        if trace.types[i] == SWC_SOMA:
            continue
        values = [float(radii[i])]
        p = int(trace.parents[i])
        if p >= 0:
            values.append(float(radii[p]))
        values.extend(float(radii[c]) for c in children[i])
        smoothed[i] = float(np.median(values))

    return dataclasses.replace(trace, radii=smoothed)


def add_branch(
    trace: TraceResult,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    start_grid: tuple[int, int, int],
    end_grid: tuple[int, int, int],
    *,
    gap_penalty: float = 50.0,
) -> TraceResult:
    """Añade una rama conectada al esqueleto entre dos puntos (grid ``z,y,x``).

    El trazado se obtiene con el camino de mínimo coste de scikit-image: coste
    ``1`` dentro de la máscara y ``gap_penalty`` fuera (permite salvar huecos
    pequeños sin abandonar la neurita). El extremo inicial se engancha al nodo
    existente más cercano; los radios salen de la transformada de distancia.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask debe ser 3D (Z, Y, X), se recibió {mask.shape}.")
    if trace.n_nodes == 0:
        raise ValueError("No hay esqueleto al que conectar la rama.")

    shape = np.array(mask.shape, dtype=np.int64)

    def _clamp(point) -> tuple[int, int, int]:
        p = np.clip(np.asarray(point, dtype=np.int64), 0, shape - 1)
        return int(p[0]), int(p[1]), int(p[2])

    start = _clamp(start_grid)
    end = _clamp(end_grid)

    cost = np.where(mask, 1.0, float(gap_penalty)).astype(np.float64)
    path, _ = route_through_array(
        cost, start, end, fully_connected=True, geometric=True
    )
    path = np.asarray(path, dtype=np.int64)
    if path.size == 0:
        raise ValueError("No se encontró un camino entre los puntos indicados.")

    spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    edt = distance_transform_edt(mask, sampling=spacing)
    new_radii = np.array([edt[z, y, x] for z, y, x in path], dtype=np.float64)

    # Nodo existente más cercano (distancia física) para enganchar el inicio.
    world = to_world(trace.coords, spacing)
    start_world = to_world(path[:1], spacing)[0]
    anchor = int(np.argmin(np.linalg.norm(world - start_world, axis=1)))

    offset = trace.n_nodes
    new_parents = np.empty(len(path), dtype=np.int64)
    new_parents[0] = anchor
    if len(path) > 1:
        new_parents[1:] = np.arange(offset, offset + len(path) - 1)

    anchor_type = int(trace.types[anchor])
    if anchor_type == SWC_SOMA:
        anchor_type = SWC_DENDRITE
    new_types = np.full(len(path), anchor_type, dtype=np.int64)

    coords = np.concatenate([trace.coords, path.astype(np.float64)], axis=0)
    parents = np.concatenate([trace.parents, new_parents])
    radii = np.concatenate([trace.radii, new_radii])
    types = np.concatenate([trace.types, new_types])
    return _rebuild(trace, coords, parents, radii, types)


def reroot_trace(trace: TraceResult, node_index: int) -> TraceResult:
    """Reenraiza la componente de ``node_index`` (el resto del bosque no cambia).

    Recorre el árbol como grafo no dirigido desde el nuevo nodo raíz, de modo
    que las ramas y su jerarquía queden orientadas desde el soma.
    """
    n = trace.n_nodes
    if not 0 <= int(node_index) < n:
        raise ValueError(f"Nodo raíz {node_index} fuera de rango [0, {n}).")

    neighbours: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        p = int(trace.parents[i])
        if p >= 0:
            neighbours[i].append(p)
            neighbours[p].append(i)

    parents = trace.parents.copy()
    visited = np.zeros(n, dtype=bool)
    root = int(node_index)
    visited[root] = True
    parents[root] = -1
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nb in neighbours[node]:
            if not visited[nb]:
                visited[nb] = True
                parents[nb] = node
                queue.append(nb)

    return _rebuild(trace, trace.coords, parents, trace.radii, trace.types)


def clear_soma(trace: TraceResult) -> TraceResult:
    """Quita el soma (elipsoide) y devuelve los nodos ``soma`` a dendrita."""
    types = trace.types.copy()
    types[types == SWC_SOMA] = SWC_DENDRITE
    return dataclasses.replace(
        trace,
        types=types,
        soma_center_um=None,
        soma_radii_um=None,
        soma_axes=None,
    )


def set_soma_from_seed(
    trace: TraceResult,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    seed_grid: tuple[int, int, int],
    *,
    radius_scale: float = 1.0,
    search_um: float = 20.0,
) -> TraceResult:
    """Designa el soma a partir de una semilla (grid ``z, y, x``).

    Busca el pico de la transformada de distancia dentro de ``search_um`` de la
    semilla (el centro del núcleo), fija ahí una esfera y reenraiza el árbol en
    el nodo del esqueleto más cercano. ``radius_scale`` permite ajustar el radio
    a mano (0.5–2.0 típicamente).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask debe ser 3D (Z, Y, X), se recibió {mask.shape}.")
    if mask.sum() == 0:
        raise ValueError("La máscara está vacía; no hay soma que detectar.")
    if trace.n_nodes == 0:
        raise ValueError("No hay esqueleto para asignar el soma.")

    spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    edt = distance_transform_edt(mask, sampling=spacing)

    shape = np.array(mask.shape, dtype=np.int64)
    seed = np.clip(np.asarray(seed_grid, dtype=np.int64), 0, shape - 1)
    radius_px = np.ceil(np.asarray(search_um) / np.asarray(spacing)).astype(np.int64)
    lo = np.maximum(seed - radius_px, 0)
    hi = np.minimum(seed + radius_px + 1, shape)
    window = edt[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    local = np.unravel_index(int(np.argmax(window)), window.shape)
    center = np.array(
        [lo[0] + local[0], lo[1] + local[1], lo[2] + local[2]], dtype=np.int64
    )
    radius = float(edt[tuple(int(v) for v in center)])
    if radius <= 0.0:
        raise ValueError("No se encontró un soma sólido en la semilla indicada.")
    radius *= float(radius_scale)

    world = to_world(trace.coords, spacing)
    center_world = to_world(center[None, :], spacing)[0]
    root = int(np.argmin(np.linalg.norm(world - center_world, axis=1)))

    rerooted = reroot_trace(trace, root)
    types = rerooted.types.copy()
    types[types == SWC_SOMA] = SWC_DENDRITE
    types[root] = SWC_SOMA
    return dataclasses.replace(
        rerooted,
        types=types,
        soma_center_um=center_world,
        soma_radii_um=np.full(3, radius, dtype=np.float64),
        soma_axes=np.eye(3),
    )