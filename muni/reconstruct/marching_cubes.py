"""Marching Cubes implementado desde cero sobre NumPy.

Usa las tablas canónicas de Paul Bourke (generadas en :mod:`mc_tables`).
El bucle por celda es Python puro por claridad; la interpolación de vértices y
normales está vectorizada por celda. Para volúmenes muy grandes, el mismo
esquema admite un bloqueo por planos (ventana rodante en Z) sin cambios de
resultado.
"""

from __future__ import annotations

import numpy as np

from muni.core.meshdata import MeshData
from muni.reconstruct.extractor import SurfaceExtractor, SurfaceExtractionResult
from muni.reconstruct.mc_tables import EDGE_TABLE, TRI_TABLE

# Vértices del cubo en orden 0..7, en coordenadas (z, y, x) relativas.
CORNER_OFFSETS = np.array(
    [
        (0, 0, 0),  # v0
        (0, 0, 1),  # v1
        (0, 1, 1),  # v2
        (0, 1, 0),  # v3
        (1, 0, 0),  # v4
        (1, 0, 1),  # v5
        (1, 1, 1),  # v6
        (1, 1, 0),  # v7
    ],
    dtype=np.int16,
)

# Cada arista conecta dos vértices del cubo (orden canónico 0..11).
EDGE_CORNERS = (
    (0, 1), (1, 2), (2, 3), (3, 0),   # aristas inferiores
    (4, 5), (5, 6), (6, 7), (7, 4),   # aristas superiores
    (0, 4), (1, 5), (2, 6), (3, 7),   # aristas verticales
)


def _gradient(field: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    """Gradiente físico (central) del campo; en los bordes usa diferencia lateral."""
    dz, dy, dx = spacing
    gz = np.gradient(field, dz, axis=0)
    gy = np.gradient(field, dy, axis=1)
    gx = np.gradient(field, dx, axis=2)
    return gx, gy, gz


class MarchingCubes(SurfaceExtractor):
    """Extracción de isosuperficie con Marching Cubes clásico."""

    name = "marching_cubes"

    def __init__(self, inside_high: bool = True) -> None:
        # inside_high=True: "interior" donde value >= isovalue (campo de probabilidad).
        self.inside_high = inside_high

    def extract(
        self,
        field: np.ndarray,
        isovalue: float = 0.5,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> SurfaceExtractionResult:
        f = np.asarray(field, dtype=np.float32)
        if f.ndim != 3:
            raise ValueError(f"Marching Cubes requiere campo 3D (Z,Y,X), se recibió {f.shape}.")
        nz, ny, nx = f.shape
        if min(nz, ny, nx) < 2:
            raise ValueError("El campo debe medir al menos 2 en cada eje.")

        iso = float(isovalue)
        gx, gy, gz = _gradient(f, spacing)
        sign = -1.0 if self.inside_high else 1.0
        sz, sy, sx = (float(s) for s in spacing)  # (µm/plano, µm/px, µm/px)

        verts: list[np.ndarray] = []
        norms: list[np.ndarray] = []

        # Se procesa por bloque de Z: el grid 2D (Y,X) de celdas se vectoriza y
        # solo se itera en Python sobre las celdas activas (de la superficie).
        for z in range(nz - 1):
            yy, xx = np.mgrid[0 : ny - 1, 0 : nx - 1]
            czv = z + CORNER_OFFSETS[:, 0]  # (8,)
            cyv = yy[..., None] + CORNER_OFFSETS[:, 1]  # (ny-1, nx-1, 8)
            cxv = xx[..., None] + CORNER_OFFSETS[:, 2]
            v = f[czv, cyv, cxv]  # (ny-1, nx-1, 8)

            inside = v >= iso if self.inside_high else v <= iso
            cubeindex = np.sum(
                inside.astype(np.uint8) << np.arange(8, dtype=np.uint8), axis=-1
            )
            edge_mask = np.take(EDGE_TABLE, cubeindex)  # (ny-1, nx-1)
            actives = np.argwhere(edge_mask != 0)  # (K, 2): fila y, col x

            for y, x in actives:
                y = int(y)
                x = int(x)
                ci = int(cubeindex[y, x])
                vc = v[y, x]  # (8,)

                # Coordenadas escalares de los 8 vértices de esta celda.
                cz = z + CORNER_OFFSETS[:, 0]
                cy = y + CORNER_OFFSETS[:, 1]
                cx = x + CORNER_OFFSETS[:, 2]

                edge_verts: dict[int, np.ndarray] = {}
                edge_norms: dict[int, np.ndarray] = {}
                for e, (a, b) in enumerate(EDGE_CORNERS):
                    if not (edge_mask[y, x] & (1 << e)):
                        continue
                    va, vb = float(vc[a]), float(vc[b])
                    denom = vb - va
                    mu = 0.0 if denom == 0 else (iso - va) / denom
                    mu = float(np.clip(mu, 0.0, 1.0))

                    pos = np.array(
                        [
                            (cz[a] + mu * (cz[b] - cz[a])) * sz,
                            (cy[a] + mu * (cy[b] - cy[a])) * sy,
                            (cx[a] + mu * (cx[b] - cx[a])) * sx,
                        ],
                        dtype=np.float32,
                    )
                    # Normal = gradiente interpolado, orientado hacia fuera.
                    grad = sign * np.array(
                        [
                            gz[cz[a], cy[a], cx[a]]
                            + mu * (gz[cz[b], cy[b], cx[b]] - gz[cz[a], cy[a], cx[a]]),
                            gy[cz[a], cy[a], cx[a]]
                            + mu * (gy[cz[b], cy[b], cx[b]] - gy[cz[a], cy[a], cx[a]]),
                            gx[cz[a], cy[a], cx[a]]
                            + mu * (gx[cz[b], cy[b], cx[b]] - gx[cz[a], cy[a], cx[a]]),
                        ],
                        dtype=np.float32,
                    )
                    edge_verts[e] = pos
                    edge_norms[e] = grad

                # Emitir triángulos según la tabla.
                tri = TRI_TABLE[ci]
                for i in range(0, 16, 3):
                    if tri[i] == -1:
                        break
                    for e in (tri[i], tri[i + 1], tri[i + 2]):
                        verts.append(edge_verts[e])
                        norms.append(edge_norms[e])
        if not verts:
            mesh = MeshData(
                vertices=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )
        else:
            soup = np.stack(verts).reshape(-1, 3, 3)
            soup_normals = np.stack(norms).reshape(-1, 3)
            mesh = MeshData.from_triangle_soup(soup, weld=True, vertex_normals=soup_normals)

        return SurfaceExtractionResult(mesh=mesh, isovalue=iso, method=self.name)
