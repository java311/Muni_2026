"""Dual Contouring implementado desde cero (sin librerías de reconstrucción).

A diferencia de Marching Cubes, Dual Contouring coloca un vértice **por celda**
(minimizando un QEF sobre los planos tangentes de las aristas intersectadas) y
une los vértices de las celdas adyacentes con quads. Captura mejor estructuras
delgadas (dendritas) y reduce ambigüedades topológicas.

Optimización: la detección de celdas y aristas activas está vectorizada con
NumPy; el bucle Python solo recorre la superficie.
"""

from __future__ import annotations

import numpy as np

from muni.core.meshdata import MeshData
from muni.reconstruct.extractor import SurfaceExtractionResult, SurfaceExtractor
from muni.reconstruct.marching_cubes import CORNER_OFFSETS, EDGE_CORNERS, _gradient


def _solve_qef(points: np.ndarray, normals: np.ndarray, max_offset: float = 5.0) -> np.ndarray:
    """Minimiza ``sum_i (n_i·(p - p_i))^2`` por mínimos cuadrados.

    - Si el sistema es singular (pocas aristas), cae a la media de los puntos.
    - Se añade una regularización leve y se limita la distancia al centroide
      (los datos de Hermite casi colineales pueden producir soluciones enormes).
    """
    if points.shape[0] == 0:
        return points.mean(axis=0)

    ata = normals.T @ normals
    trace = float(np.trace(ata))
    if trace > 0:
        ata = ata + np.eye(3, dtype=np.float64) * (1e-6 * trace)
    atb = normals.T @ np.einsum("ij,ij->i", normals, points)
    centroid = points.mean(axis=0)

    try:
        p = np.linalg.solve(ata, atb)
        if not np.all(np.isfinite(p)):
            raise np.linalg.LinAlgError
        if float(np.linalg.norm(p - centroid)) > max_offset:
            return centroid.astype(np.float32)
        return p.astype(np.float32)
    except np.linalg.LinAlgError:
        return centroid.astype(np.float32)


def _trilinear_interp(arr: np.ndarray, pz: float, py: float, px: float) -> float:
    """Interpolación trilineal de un array 3D en coordenadas continuas (z, y, x)."""
    z0 = max(0, min(int(pz), arr.shape[0] - 2))
    y0 = max(0, min(int(py), arr.shape[1] - 2))
    x0 = max(0, min(int(px), arr.shape[2] - 2))
    fz = min(max(pz - z0, 0.0), 1.0)
    fy = min(max(py - y0, 0.0), 1.0)
    fx = min(max(px - x0, 0.0), 1.0)
    c000 = arr[z0, y0, x0]
    c001 = arr[z0, y0, x0 + 1]
    c010 = arr[z0, y0 + 1, x0]
    c011 = arr[z0, y0 + 1, x0 + 1]
    c100 = arr[z0 + 1, y0, x0]
    c101 = arr[z0 + 1, y0, x0 + 1]
    c110 = arr[z0 + 1, y0 + 1, x0]
    c111 = arr[z0 + 1, y0 + 1, x0 + 1]
    return (
        c000 * (1 - fz) * (1 - fy) * (1 - fx)
        + c001 * (1 - fz) * (1 - fy) * fx
        + c010 * (1 - fz) * fy * (1 - fx)
        + c011 * (1 - fz) * fy * fx
        + c100 * fz * (1 - fy) * (1 - fx)
        + c101 * fz * (1 - fy) * fx
        + c110 * fz * fy * (1 - fx)
        + c111 * fz * fy * fx
    )


class DualContouring(SurfaceExtractor):
    """Extracción de isosuperficie con Dual Contouring."""

    name = "dual_contouring"

    def __init__(self, inside_high: bool = True) -> None:
        self.inside_high = inside_high

    def extract(
        self,
        field: np.ndarray,
        isovalue: float = 0.5,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> SurfaceExtractionResult:
        f = np.asarray(field, dtype=np.float32)
        if f.ndim != 3:
            raise ValueError(f"Dual Contouring requiere campo 3D (Z,Y,X), se recibió {f.shape}.")
        nz, ny, nx = f.shape
        if min(nz, ny, nx) < 2:
            raise ValueError("El campo debe medir al menos 2 en cada eje.")

        iso = float(isovalue)
        gx, gy, gz = _gradient(f, spacing)
        sign = -1.0 if self.inside_high else 1.0
        sz, sy, sx = (float(s) for s in spacing)  # (µm/plano, µm/px, µm/px)

        def inside(val: np.ndarray) -> np.ndarray:
            return val >= iso if self.inside_high else val <= iso

        ins = inside(f)  # máscara booleana completa (nz, ny, nx)

        # --- 1) vértice dual de cada celda activa (por bloque de Z, vectorizado) ---
        cell_shape = (nz - 1, ny - 1, nx - 1)
        cell_id = np.full(cell_shape, -1, dtype=np.int32)
        vertices: list[np.ndarray] = []
        normals: list[np.ndarray] = []

        for z in range(nz - 1):
            yy, xx = np.mgrid[0 : ny - 1, 0 : nx - 1]
            cz = CORNER_OFFSETS[:, 0]  # (8,)
            cy = yy[..., None] + CORNER_OFFSETS[:, 1]
            cx = xx[..., None] + CORNER_OFFSETS[:, 2]
            v = f[z + cz, cy, cx]  # (ny-1, nx-1, 8)
            inn = ins[z + cz, cy, cx]
            active = ~(inn.all(axis=-1) | ~inn.any(axis=-1))  # hay cambio de signo
            actives = np.argwhere(active)  # (K, 2): y, x

            for y, x in actives:
                y = int(y)
                x = int(x)
                pts: list[np.ndarray] = []
                nrm: list[np.ndarray] = []
                czc = z + CORNER_OFFSETS[:, 0]
                cyc = y + CORNER_OFFSETS[:, 1]
                cxc = x + CORNER_OFFSETS[:, 2]
                vc = v[y, x]
                inc = inn[y, x]

                for e, (a, b) in enumerate(EDGE_CORNERS):
                    if inc[a] == inc[b]:
                        continue
                    va, vb = float(vc[a]), float(vc[b])
                    denom = vb - va
                    mu = 0.0 if denom == 0 else (iso - va) / denom
                    mu = float(np.clip(mu, 0.0, 1.0))
                    pos = np.array(
                        [
                            (cxc[a] + mu * (cxc[b] - cxc[a])) * sx,
                            (cyc[a] + mu * (cyc[b] - cyc[a])) * sy,
                            (czc[a] + mu * (czc[b] - czc[a])) * sz,
                        ],
                        dtype=np.float32,
                    )
                    grad = np.array(
                        [
                            gx[czc[a], cyc[a], cxc[a]]
                            + mu * (gx[czc[b], cyc[b], cxc[b]] - gx[czc[a], cyc[a], cxc[a]]),
                            gy[czc[a], cyc[a], cxc[a]]
                            + mu * (gy[czc[b], cyc[b], cxc[b]] - gy[czc[a], cyc[a], cxc[a]]),
                            gz[czc[a], cyc[a], cxc[a]]
                            + mu * (gz[czc[b], cyc[b], cxc[b]] - gz[czc[a], cyc[a], cxc[a]]),
                        ],
                        dtype=np.float32,
                    )
                    pts.append(pos)
                    nrm.append(grad)

                if not pts:
                    continue
                pts_a = np.stack(pts)
                nrm_a = np.stack(nrm)
                vertex = _solve_qef(pts_a, nrm_a)

                vz = vertex[2] / sz
                vy = vertex[1] / sy
                vx = vertex[0] / sx
                normal = sign * np.array(
                    [
                        _trilinear_interp(gx, vz, vy, vx),
                        _trilinear_interp(gy, vz, vy, vx),
                        _trilinear_interp(gz, vz, vy, vx),
                    ],
                    dtype=np.float32,
                )
                ln = np.linalg.norm(normal)
                if ln > 0:
                    normal = normal / ln

                cell_id[z, y, x] = len(vertices)
                vertices.append(vertex)
                normals.append(normal)

        if not vertices:
            empty = MeshData(
                vertices=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )
            return SurfaceExtractionResult(mesh=empty, isovalue=iso, method=self.name)

        # --- 2) quads: vértices de las 4 celdas alrededor de cada arista activa ---
        def cell_at(cz: int, cy: int, cx: int) -> int:
            if 0 <= cz < nz - 1 and 0 <= cy < ny - 1 and 0 <= cx < nx - 1:
                return int(cell_id[cz, cy, cx])
            return -1

        def add_quad(c0: int, c1: int, c2: int, c3: int, faces: list[list[int]]) -> None:
            ids = [c0, c1, c2, c3]
            if any(i < 0 for i in ids) or len(set(ids)) != 4:
                return
            faces.append([ids[0], ids[1], ids[2]])
            faces.append([ids[0], ids[2], ids[3]])

        faces: list[list[int]] = []

        # Aristas paralelas a X (cambio de signo entre (z,y,x) y (z,y,x+1)).
        cross_x = np.argwhere(ins[:, :, :-1] != ins[:, :, 1:])  # (Kx, 3): z,y,x
        for z, y, x in cross_x:
            z, y, x = int(z), int(y), int(x)
            add_quad(
                cell_at(z - 1, y - 1, x),
                cell_at(z - 1, y, x),
                cell_at(z, y, x),
                cell_at(z, y - 1, x),
                faces,
            )

        # Aristas paralelas a Y.
        cross_y = np.argwhere(ins[:, :-1, :] != ins[:, 1:, :])
        for z, y, x in cross_y:
            z, y, x = int(z), int(y), int(x)
            add_quad(
                cell_at(z - 1, y, x - 1),
                cell_at(z - 1, y, x),
                cell_at(z, y, x),
                cell_at(z, y, x - 1),
                faces,
            )

        # Aristas paralelas a Z.
        cross_z = np.argwhere(ins[:-1, :, :] != ins[1:, :, :])
        for z, y, x in cross_z:
            z, y, x = int(z), int(y), int(x)
            add_quad(
                cell_at(z, y - 1, x - 1),
                cell_at(z, y - 1, x),
                cell_at(z, y, x),
                cell_at(z, y, x - 1),
                faces,
            )

        if not faces:
            empty = MeshData(
                vertices=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )
            return SurfaceExtractionResult(mesh=empty, isovalue=iso, method=self.name)

        faces_arr = np.asarray(faces, dtype=np.int32)
        a, b, c = faces_arr[:, 0], faces_arr[:, 1], faces_arr[:, 2]
        keep = (a != b) & (b != c) & (a != c)
        faces_arr = faces_arr[keep]

        mesh = MeshData(
            vertices=np.asarray(vertices, dtype=np.float32),
            normals=np.asarray(normals, dtype=np.float32),
            faces=faces_arr,
        )
        return SurfaceExtractionResult(mesh=mesh, isovalue=iso, method=self.name)
