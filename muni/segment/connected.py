"""Etiquetado de componentes conexos desde cero (sin scipy).

La implementación es correcta y prioriza claridad; para volúmenes muy grandes
puede ser lenta (se itera con BFS en Python). Es una utilidad auxiliar, no el
camino crítico de la reconstrucción.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Etiqueta los componentes conexos (vecindad 6 en 3D, 4 en 2D).

    Returns
    -------
    (labels, count):
        ``labels`` con 0 para fondo y 1..N para cada componente.
    """
    m = np.asarray(mask, dtype=bool)
    labels = np.zeros(m.shape, dtype=np.int32)
    if not m.any():
        return labels, 0

    n_comp = 0
    idx = np.flatnonzero(m)
    # Índices 1D ya visitados
    visited = np.zeros(m.size, dtype=bool)
    for start in idx:
        if visited[start]:
            continue
        n_comp += 1
        queue = deque([start])
        visited[start] = True
        labels.flat[start] = n_comp
        while queue:
            cur = queue.popleft()
            coords = np.unravel_index(cur, m.shape)
            for axis in range(m.ndim):
                for step in (-1, 1):
                    n = list(coords)
                    n[axis] += step
                    if not 0 <= n[axis] < m.shape[axis]:
                        continue
                    nidx = int(np.ravel_multi_index(n, m.shape))
                    if m.flat[nidx] and not visited[nidx]:
                        visited[nidx] = True
                        labels.flat[nidx] = n_comp
                        queue.append(nidx)
    return labels, n_comp


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Devuelve una máscara booleana con solo el componente conexo más grande."""
    labels, n_comp = connected_components(mask)
    if n_comp == 0:
        return np.asarray(mask, dtype=bool).copy()
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # el fondo no cuenta
    biggest = int(np.argmax(counts))
    return labels == biggest
