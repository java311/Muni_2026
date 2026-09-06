"""Escritor y lector de archivos SWC (formato estándar de morfología neuronal)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from muni.trace.base import TraceResult


def write_swc(
    path: str | Path,
    result: TraceResult,
    *,
    spacing: tuple[float, float, float] | None = None,
    swc_type: int = 3,
) -> Path:
    """Escribe un archivo SWC a partir de un TraceResult.

    Parameters
    ----------
    path:
        Ruta del archivo de salida (se añade `.swc` si no lo tiene).
    result:
        Resultado del tracing.
    spacing:
        Espaciado ``(dz, dy, dx)`` en micras. Si es None, usa
        ``result.spacing``.
    swc_type:
        Tipo SWC por defecto (3 = dendrita).

    Returns
    -------
    Ruta completa del archivo escrito.
    """
    path = Path(path)
    if path.suffix != ".swc":
        path = path.with_suffix(".swc")

    if spacing is None:
        spacing = result.spacing

    n = result.n_nodes
    ids = np.arange(1, n + 1, dtype=np.int64)
    types = np.full(n, swc_type, dtype=np.int64)

    parents = result.parents.copy()
    parents[parents != -1] += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# spacing {spacing[0]} {spacing[1]} {spacing[2]}\n")
        f.write("# id type x y z radius parent\n")
        for i in range(n):
            x, y, z = result.coords[i, 2], result.coords[i, 1], result.coords[i, 0]
            r = result.radii[i]
            p = int(parents[i])
            f.write(f"{ids[i]} {types[i]} {x:.4f} {y:.4f} {z:.4f} {r:.4f} {p}\n")

    return path


def load_swc(path: str | Path) -> tuple[TraceResult, tuple[float, float, float]]:
    """Lee un archivo SWC y devuelve (TraceResult, spacing).

    Returns
    -------
    trace:
        Resultado del tracing.
    spacing:
        Espaciado ``(dz, dy, dx)`` en micras (1.0 si no estaba en el archivo).
    """
    path = Path(path)
    coords = []
    types = []
    radii = []
    parents_raw = []
    spacing = (1.0, 1.0, 1.0)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("# spacing"):
                    parts = line.split()
                    if len(parts) == 5:
                        spacing = (float(parts[2]), float(parts[3]), float(parts[4]))
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            coords.append([float(parts[4]), float(parts[3]), float(parts[2])])  # z, y, x
            types.append(int(parts[1]))
            radii.append(float(parts[5]))
            parents_raw.append(int(parts[6]))

    n = len(coords)
    if n == 0:
        return TraceResult(
            coords=np.empty((0, 3)),
            radii=np.empty(0),
            parents=np.empty(0, dtype=np.int64),
            branch_labels=np.empty(0, dtype=np.int64),
            total_length_um=0.0,
            spacing=spacing,
            method="swc_import",
        ), spacing

    # Convertir de 1-based a 0-based.
    parents = np.array(parents_raw, dtype=np.int64)
    parents[parents > 0] -= 1
    parents[parents < -1] = -1

    # Calcular longitud total.
    coords_arr = np.array(coords, dtype=np.float64)
    radii_arr = np.array(radii, dtype=np.float64)
    total_length = 0.0
    for i in range(n):
        p = int(parents[i])
        if p >= 0:
            dz = (coords_arr[i, 0] - coords_arr[p, 0]) * spacing[0]
            dy = (coords_arr[i, 1] - coords_arr[p, 1]) * spacing[1]
            dx = (coords_arr[i, 2] - coords_arr[p, 2]) * spacing[2]
            total_length += np.sqrt(dz * dz + dy * dy + dx * dx)

    return TraceResult(
        coords=coords_arr,
        radii=radii_arr,
        parents=parents,
        branch_labels=np.zeros(n, dtype=np.int64),
        total_length_um=total_length,
        spacing=spacing,
        method="swc_import",
    ), spacing
