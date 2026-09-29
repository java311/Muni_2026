"""Pruebas de la malla de tubos generada desde el esqueleto."""

import numpy as np

from muni.reconstruct.tube_mesh import tube_mesh_from_trace
from muni.trace.base import SWC_DENDRITE, TraceResult


def _straight_trace(n=6, radius=1.0, soma=False):
    coords = np.stack(
        [np.zeros(n), np.zeros(n), np.arange(n, dtype=float)], axis=1
    )  # (z, y, x)
    parents = np.concatenate([[-1], np.arange(n - 1)])
    radii = np.full(n, radius)
    kwargs = {}
    if soma:
        kwargs.update(
            soma_center_um=np.array([2.5, 0.0, 0.0]),
            soma_radii_um=np.array([2.0, 2.0, 2.0]),
            soma_axes=np.eye(3),
        )
    return TraceResult(
        coords=coords,
        radii=radii,
        parents=parents.astype(np.int64),
        branch_labels=np.zeros(n, dtype=np.int64),
        total_length_um=float(n - 1),
        spacing=(1.0, 1.0, 1.0),
        types=np.full(n, SWC_DENDRITE, dtype=np.int64),
        **kwargs,
    )


def test_tube_mesh_is_valid_and_matches_radius():
    trace = _straight_trace(radius=1.0, soma=True)
    mesh = tube_mesh_from_trace(trace, sides=8)

    assert not mesh.is_empty
    assert mesh.validate() == []
    lo, hi = mesh.bounds()
    # Tubo de radio 1 centrado en el eje X, soma esférico de radio 2 en x≈2.5.
    assert float(np.max(np.abs(lo[1:]))) <= 2.0 + 1e-3
    assert float(np.max(np.abs(hi[1:]))) <= 2.0 + 1e-3
    assert float(lo[0]) <= 1e-3 and float(hi[0]) >= 5.0 - 1e-3


def test_tube_mesh_without_soma_stays_thin():
    trace = _straight_trace(radius=1.0, soma=False)
    mesh = tube_mesh_from_trace(trace, sides=8)

    assert not mesh.is_empty
    assert mesh.validate() == []
    lo, hi = mesh.bounds()
    assert float(np.max(np.abs(lo[1:]))) <= 1.0 + 1e-3
    assert float(np.max(np.abs(hi[1:]))) <= 1.0 + 1e-3


def test_empty_trace_gives_empty_mesh():
    trace = TraceResult(
        coords=np.empty((0, 3)),
        radii=np.empty(0),
        parents=np.empty(0, dtype=np.int64),
        branch_labels=np.empty(0, dtype=np.int64),
        total_length_um=0.0,
        spacing=(1.0, 1.0, 1.0),
    )
    assert tube_mesh_from_trace(trace).is_empty