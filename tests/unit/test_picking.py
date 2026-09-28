"""Pruebas de la selección de nodos del esqueleto (proyección a pantalla)."""

import numpy as np

from muni.trace.base import TraceResult
from muni.view.camera import OrbitCamera
from muni.view.picking import nearest_node


def _trace(points):
    coords = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(coords)
    parents = (
        np.concatenate([[-1], np.arange(n - 1)]).astype(np.int64)
        if n
        else np.empty(0, dtype=np.int64)
    )
    return TraceResult(
        coords=coords,
        radii=np.ones(n),
        parents=parents,
        branch_labels=np.zeros(n, dtype=np.int64),
        total_length_um=0.0,
        spacing=(1.0, 1.0, 1.0),
    )


def test_nearest_node_center_hit():
    cam = OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0)
    trace = _trace([[0, 0, 0], [0, 0, 50]])  # mundo (x=0) y (x=50)
    idx = nearest_node(trace, cam, 200.0, 200.0, 400, 400)
    assert idx == 0


def test_nearest_node_far_click_returns_none():
    cam = OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0)
    trace = _trace([[0, 0, 0]])
    assert nearest_node(trace, cam, 5.0, 5.0, 400, 400) is None


def test_nearest_node_empty_trace():
    cam = OrbitCamera()
    assert nearest_node(_trace([]), cam, 200, 200, 400, 400) is None
    assert nearest_node(None, cam, 200, 200, 400, 400) is None