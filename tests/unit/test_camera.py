"""Pruebas de la cámara orbital (F5, lógica pura sin Qt)."""

import numpy as np
import pytest

from muni.view.camera import OrbitCamera


def test_eye_and_view_center():
    cam = OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0, yaw=-45.0, pitch=25.0)
    eye = cam.eye()
    assert np.linalg.norm(eye) == pytest.approx(100.0)

    # El centro del mundo debe proyectarse al centro de pantalla (NDC ≈ 0,0).
    ndc = cam.project(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert abs(ndc[0]) < 1e-4 and abs(ndc[1]) < 1e-4


def test_ray_through_center_hits_target():
    cam = OrbitCamera(target=(5.0, -2.0, 3.0), distance=80.0)
    origin, direction = cam.ray(0.0, 0.0)
    # El rayo central debe apuntar prácticamente hacia el objetivo.
    to_target = cam.target - origin
    to_target = to_target / np.linalg.norm(to_target)
    assert np.dot(direction, to_target) > 0.999


def test_fit_centers_and_keeps_distance():
    cam = OrbitCamera()
    cam.fit(np.array([0.0, 0.0, 0.0]), np.array([10.0, 20.0, 30.0]))
    assert np.allclose(cam.target, [5.0, 10.0, 15.0], atol=1e-4)
    assert cam.distance > 0
    # El primer vértice debe quedar dentro del frustum (z NDC entre -1 y 1).
    ndc = cam.project(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert -1.0 <= ndc[2] <= 1.0


def test_zoom_and_pan_bounds():
    cam = OrbitCamera(distance=100.0)
    cam.zoom(2.0)
    assert cam.distance == pytest.approx(200.0)
    before = cam.target.copy()
    cam.pan(10.0, 0.0, 600.0)
    assert not np.allclose(cam.target, before)