"""Tracing dendrítico usando scikit-image (skeletonize_3d) y skan."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from muni.trace.base import Tracer, TraceResult


def _build_adjacency(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    voxel_set = set(idx.tolist())
    voxel_to_node = {v: i for i, v in enumerate(idx.tolist())}

    neighbors_26 = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                neighbors_26.append((dz, dy, dx))

    adj_list = np.empty(n, dtype=object)
    degree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        adj_list[i] = []

    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in neighbors_26:
            nz, ny, nx = z + dz, y + dy, x + dx
            v = nz * max_dim * max_dim + ny * max_dim + nx
            if v in voxel_set:
                j = voxel_to_node[v]
                adj_list[i].append(j)
                degree[i] += 1

    return coords, adj_list, degree


class SkimageTracer(Tracer):
    """Tracing usando skeletonize_3d (Lee94) + skan para análisis."""

    name = "skimage"

    def trace(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> TraceResult:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3:
            raise ValueError(f"mask debe ser 3D, se recibió forma {mask.shape}.")
        if mask.sum() == 0:
            raise ValueError("mask está vacía (todos False).")

        # 1. Esqueleto.
        skeleton = skeletonize(mask)

        # 2. Distance transform para radios.
        edt = distance_transform_edt(mask, sampling=spacing)

        # 3. Construir grafo.
        coords_grid, adj_list, degree = _build_adjacency(skeleton)
        n = len(coords_grid)
        if n == 0:
            return TraceResult(
                coords=np.empty((0, 3)),
                radii=np.empty(0),
                parents=np.empty(0, dtype=np.int64),
                branch_labels=np.empty(0, dtype=np.int64),
                total_length_um=0.0,
                spacing=tuple(spacing),
                method=self.name,
            )

        # 4. Radios sobre el esqueleto.
        radii = np.array([edt[z, y, x] for z, y, x in coords_grid], dtype=np.float64)

        # 5. Seleccionar raíz: punto final con radio más grande (soma aproximado).
        endpoints = np.where(degree == 1)[0]
        if len(endpoints) == 0:
            root = int(np.argmax(radii))
        else:
            root = int(endpoints[np.argmax(radii[endpoints])])

        # 6. BFS para construir parents.
        parents = np.full(n, -1, dtype=np.int64)
        visited = np.zeros(n, dtype=bool)
        visited[root] = True
        queue = deque([root])

        while queue:
            node = queue.popleft()
            for nb in adj_list[node]:
                if not visited[nb]:
                    visited[nb] = True
                    parents[nb] = node
                    queue.append(nb)

        # 7. Asignar etiquetas de rama.
        # Una nueva rama empieza en cada nodo de ramificación (grado >= 3).
        branch_labels = np.zeros(n, dtype=np.int64)
        branch_counter = 0

        # Encontrar raíces de subárboles: nodos cuyo padre es un branch point.
        child_of_branch_point = []
        for i in range(n):
            p = parents[i]
            if p == -1:
                continue
            if degree[p] >= 3:
                child_of_branch_point.append(i)

        # BFS desde cada hijo de branch point para etiquetar ramas.
        labeled = np.zeros(n, dtype=bool)
        labeled[root] = True
        # Marcar nodos del tronco principal.
        node = root
        while parents[node] != -1 or degree[node] >= 2:
            # Seguir el tronco hasta un branch point.
            next_nodes = [nb for nb in adj_list[node] if parents[nb] == node and not labeled[nb]]
            if len(next_nodes) == 0:
                break
            if len(next_nodes) == 1:
                labeled[node] = True
                node = next_nodes[0]
                labeled[node] = True
            else:
                # Branch point encontrado.
                break

        # Etiquetar cada subárbol que empieza en un branch point.
        for child_root in child_of_branch_point:
            if branch_labels[child_root] != 0:
                continue
            branch_counter += 1
            q = deque([child_root])
            branch_labels[child_root] = branch_counter
            while q:
                nd = q.popleft()
                for nb in adj_list[nd]:
                    if parents[nb] == nd and branch_labels[nb] == 0:
                        branch_labels[nb] = branch_counter
                        q.append(nb)

        # 8. Longitud total.
        total_length = 0.0
        for i in range(n):
            p = parents[i]
            if p != -1:
                dz = (coords_grid[i, 0] - coords_grid[p, 0]) * spacing[0]
                dy = (coords_grid[i, 1] - coords_grid[p, 1]) * spacing[1]
                dx = (coords_grid[i, 2] - coords_grid[p, 2]) * spacing[2]
                total_length += np.sqrt(dz * dz + dy * dy + dx * dx)

        return TraceResult(
            coords=coords_grid.astype(np.float64),
            radii=radii,
            parents=parents,
            branch_labels=branch_labels,
            total_length_um=total_length,
            spacing=tuple(spacing),
            method=self.name,
        )
