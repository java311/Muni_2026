"""Utilidades de grafo para esqueletos 3D (compartidas por tracer y edición).

Todas las funciones trabajan con coordenadas de grid ``(z, y, x)`` y espaciado
físico ``(dz, dy, dx)``, y son puras: no mutan los arrays de entrada.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from muni.trace.soma import Soma

NEIGHBORS_26 = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def build_adjacency(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye grafo de adyacencia a partir de un esqueleto binario 3D.

    Returns
    -------
    coords_grid:
        Coordenadas ``(N, 3)`` de los voxels del esqueleto en espacio de grid.
    adj_list:
        Lista de adyacencia como array de objetos.
    degree:
        Grado de cada nodo.
    """
    coords = np.argwhere(skeleton > 0)  # (N, 3) — (z, y, x)
    n = len(coords)
    if n == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=object), np.zeros(0, dtype=np.int64)

    # Indexar voxels en espacio 1D para búsqueda rápida.
    max_dim = max(coords.max(axis=0) + 1)
    idx = coords[:, 0] * max_dim * max_dim + coords[:, 1] * max_dim + coords[:, 2]
    voxel_to_node = {v: i for i, v in enumerate(idx.tolist())}

    adj_list = np.empty(n, dtype=object)
    degree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        adj_list[i] = []

    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in NEIGHBORS_26:
            v = (z + dz) * max_dim * max_dim + (y + dy) * max_dim + (x + dx)
            j = voxel_to_node.get(v)
            if j is not None:
                adj_list[i].append(j)
                degree[i] += 1

    return coords, adj_list, degree


def to_world(coords_grid: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Convierte coords de grid ``(z, y, x)`` a mundo físico ``(x, y, z)`` en µm."""
    world = np.empty((len(coords_grid), 3), dtype=np.float64)
    world[:, 0] = coords_grid[:, 2] * spacing[2]
    world[:, 1] = coords_grid[:, 1] * spacing[1]
    world[:, 2] = coords_grid[:, 0] * spacing[0]
    return world


def edge_length(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * spacing[0]
    dy = (a[1] - b[1]) * spacing[1]
    dx = (a[2] - b[2]) * spacing[2]
    return float(np.sqrt(dz * dz + dy * dy + dx * dx))


def total_length(
    coords_grid: np.ndarray, parents: np.ndarray, spacing: tuple[float, float, float]
) -> float:
    """Longitud total del bosque (suma de las aristas con padre)."""
    return float(
        sum(
            edge_length(coords_grid[i], coords_grid[int(parents[i])], spacing)
            for i in range(len(parents))
            if parents[i] >= 0
        )
    )


def build_forest(
    coords_grid: np.ndarray,
    adj_list: np.ndarray,
    degree: np.ndarray,
    radii: np.ndarray,
    node_comp: np.ndarray,
    soma: Soma | None,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Construye los padres de todos los nodos, enraizando cada componente.

    Cada componente conexo (soma o fragmento) recibe su propia raíz; así ningún
    tramo del esqueleto queda sin asignar.
    """
    n = len(coords_grid)
    parents = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    world = to_world(coords_grid, spacing)

    for comp_id in np.unique(node_comp):
        idx = np.flatnonzero(node_comp == comp_id)
        root = None
        if soma is not None:
            dist = np.linalg.norm(world[idx] - soma.center_um, axis=1)
            if dist.min() <= 4.0 * soma.radius_um:
                root = int(idx[int(np.argmin(dist))])
        if root is None:
            ends = idx[degree[idx] == 1]
            pool = ends if len(ends) else idx
            root = int(pool[int(np.argmax(radii[pool]))])

        visited[root] = True
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for nb in adj_list[node]:
                if node_comp[nb] == comp_id and not visited[nb]:
                    visited[nb] = True
                    parents[nb] = node
                    queue.append(nb)

    return parents


def children_of(parents: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for i, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(i)
    return children


def prune_spurs(
    coords_grid: np.ndarray,
    parents: np.ndarray,
    spacing: tuple[float, float, float],
    min_spur_um: float,
) -> np.ndarray:
    """Marca (``False``) las cadenas terminales más cortas que ``min_spur_um``.

    Solo se podan ramas que terminan en un punto de bifurcación; nunca se
    elimina la raíz ni el tronco principal.
    """
    n = len(parents)
    children = children_of(parents)
    child_count = np.array([len(ch) for ch in children])
    keep = np.ones(n, dtype=bool)

    for leaf in range(n):
        if child_count[leaf] != 0 or parents[leaf] < 0:
            continue
        path = [leaf]
        length = 0.0
        cur = leaf
        spur = False
        while True:
            p = int(parents[cur])
            if p < 0:
                break
            length += edge_length(coords_grid[cur], coords_grid[p], spacing)
            if child_count[p] >= 2:
                spur = True
                break
            path.append(p)
            cur = p
        if spur and length < min_spur_um:
            for node in path:
                keep[node] = False
    return keep


def compact(
    coords_grid: np.ndarray,
    parents: np.ndarray,
    radii: np.ndarray,
    node_comp: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reindexa los arrays descartando los nodos no conservados."""
    index = np.flatnonzero(keep)
    remap = -np.ones(len(parents), dtype=np.int64)
    remap[index] = np.arange(len(index), dtype=np.int64)
    new_parents = parents[index].copy()
    has_parent = new_parents >= 0
    new_parents[has_parent] = remap[new_parents[has_parent]]
    return coords_grid[index], new_parents, radii[index], node_comp[index]


def branch_labels(parents: np.ndarray, children: list[list[int]]) -> np.ndarray:
    """Reparte los nodos en ramas (cadenas entre nodos de interés)."""
    n = len(parents)
    child_count = np.array([len(ch) for ch in children])
    is_node = (child_count + (parents >= 0)) != 2
    is_node |= parents < 0

    labels = np.full(n, -1, dtype=np.int64)
    branch = 0
    for i in range(n):
        if not is_node[i]:
            continue
        for c in children[i]:
            if labels[c] != -1:
                continue
            labels[c] = branch
            cur = c
            while not is_node[cur]:
                nxt = children[cur]
                if len(nxt) != 1:
                    break
                cur = nxt[0]
                if labels[cur] == -1:
                    labels[cur] = branch
            branch += 1
    for i in range(n):
        if labels[i] == -1:
            labels[i] = branch
            branch += 1
    return labels


def branch_start(branch_idx: np.ndarray, parents: np.ndarray) -> int:
    members = set(branch_idx.tolist())
    for node in branch_idx:
        p = int(parents[node])
        if p < 0 or p not in members:
            return int(node)
    return int(branch_idx[0])


def branch_parent_map(labels: np.ndarray, parents: np.ndarray) -> dict[int, int | None]:
    """Rama padre de cada rama (``None`` si su base es una raíz del bosque)."""
    result: dict[int, int | None] = {}
    for bid in np.unique(labels):
        idx = np.flatnonzero(labels == bid)
        start = branch_start(idx, parents)
        p = int(parents[start])
        parent_bid = int(labels[p]) if p >= 0 else None
        # Una rama no puede ser su propia madre (esqueletos bien formados).
        result[int(bid)] = None if parent_bid == int(bid) else parent_bid
    return result


def branch_tree(
    labels: np.ndarray, parents: np.ndarray
) -> list[tuple[int, int | None, int]]:
    """Ordena las ramas en preorden: ``(rama, rama_padre, profundidad)``.

    Soporta bosques con varias raíces (fragmentos separados).
    """
    parent = branch_parent_map(labels, parents)
    children: dict[int | None, list[int]] = {}
    for bid, parent_bid in parent.items():
        children.setdefault(parent_bid, []).append(bid)

    ordered: list[tuple[int, int | None, int]] = []

    def walk(bid: int, parent_bid: int | None, depth: int) -> None:
        ordered.append((bid, parent_bid, depth))
        for child in sorted(children.get(bid, [])):
            walk(child, bid, depth + 1)

    for root in sorted(children.get(None, [])):
        walk(root, None, 0)
    return ordered


def branch_ancestors(labels: np.ndarray, parents: np.ndarray, bid: int) -> list[int]:
    """Cadena ``[bid, rama_madre, …, raíz]`` siguiendo el árbol hacia arriba."""
    parent = branch_parent_map(labels, parents)
    chain: list[int] = []
    seen: set[int] = set()
    cur: int | None = int(bid)
    while cur is not None and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        cur = parent.get(cur)
    return chain


def branch_descendants(labels: np.ndarray, parents: np.ndarray, bid: int) -> set[int]:
    """Rama ``bid`` y todas sus ramas hijas (el subárbol completo)."""
    parent = branch_parent_map(labels, parents)
    children: dict[int, list[int]] = {}
    for branch, parent_bid in parent.items():
        if parent_bid is not None:
            children.setdefault(parent_bid, []).append(branch)
    result: set[int] = set()
    stack = [int(bid)]
    while stack:
        branch = stack.pop()
        if branch in result:
            continue
        result.add(branch)
        stack.extend(children.get(branch, []))
    return result