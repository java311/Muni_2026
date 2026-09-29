"""Cámara orbital para el visor 3D (matemática pura, sin Qt).

Convención: matrices 4x4 en NumPy (``v_clip = P @ V @ M @ v_world``), con
almacenamiento por filas. El visor las convierte a ``QMatrix4x4`` (orden por
columnas) al pasarlas al shader.
"""

from __future__ import annotations

import numpy as np


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Matriz de vista (mundo → cámara)."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Matriz de proyección en perspectiva."""
    f = 1.0 / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


class OrbitCamera:
    """Cámara orbital: rota alrededor de un objetivo, con pan y zoom."""

    def __init__(
        self,
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 100.0,
        yaw: float = -45.0,
        pitch: float = 25.0,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 10000.0,
    ) -> None:
        self.target = np.asarray(target, dtype=np.float32)
        self.distance = float(distance)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)
        self.aspect = 1.0

    # ------------------------------------------------------------------ props
    def eye(self) -> np.ndarray:
        yaw = np.deg2rad(self.yaw)
        pitch = np.deg2rad(self.pitch)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        direction = np.array([cp * sy, sp, cp * cy], dtype=np.float32)
        return self.target + self.distance * direction

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.eye(), self.target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def projection_matrix(self) -> np.ndarray:
        return _perspective(self.fov, self.aspect, self.near, self.far)

    def set_aspect(self, width: int, height: int) -> None:
        self.aspect = width / max(height, 1)

    # -------------------------------------------------------------- interacción
    def rotate(self, dyaw: float, dpitch: float) -> None:
        self.yaw += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch, -89.0, 89.0))

    def zoom(self, factor: float) -> None:
        """``factor < 1`` acerca; ``factor > 1`` aleja."""
        self.distance = float(np.clip(self.distance * factor, 1e-3, 1e6))

    def pan(self, dx_px: float, dy_px: float, viewport_height: float) -> None:
        # Desplazamiento proporcional a la distancia y al fov.
        scale = 2.0 * self.distance * np.tan(np.deg2rad(self.fov) / 2.0) / viewport_height
        right = self._right()
        up = self._up()
        self.target = self.target + (right * (-dx_px) + up * (dy_px)) * scale

    def _right(self) -> np.ndarray:
        f = self.target - self.eye()
        f /= np.linalg.norm(f)
        right = np.cross(f, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        n = np.linalg.norm(right)
        return right / n if n else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def _up(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def fit(self, bounds_min: np.ndarray, bounds_max: np.ndarray) -> None:
        """Encuadra la caja delimitadora.

        Posiciona la cámara mirando de frente la cara más grande del modelo
        (a lo largo del eje más corto), con un ángulo de pitch suave para
        dar profundidad.
        """
        center = (
            np.asarray(bounds_min, dtype=np.float32) + np.asarray(bounds_max, dtype=np.float32)
        ) / 2.0
        size = np.asarray(bounds_max, dtype=np.float32) - np.asarray(bounds_min, dtype=np.float32)
        radius = float(np.linalg.norm(size) / 2.0) or 1.0
        self.target = center
        self.distance = radius / np.tan(np.deg2rad(self.fov) / 2.0) * 1.4
        ex, ey, ez = float(size[0]), float(size[1]), float(size[2])
        pitch_deg = 15.0
        if ez <= ex and ez <= ey:
            # Z más corto → mirar de frente por Z (cara XY).
            self.yaw = 0.0
            self.pitch = pitch_deg
        elif ey <= ex and ey <= ez:
            # Y más corto → mirar de frente por Y (cara XZ).
            self.pitch = 90.0 - pitch_deg
            self.yaw = 0.0
        else:
            # X más corto → mirar de frente por X (cara YZ).
            self.yaw = 90.0
            self.pitch = pitch_deg

    def project(self, point: np.ndarray) -> np.ndarray:
        """Proyecta un punto 3D a coordenadas normalizadas de dispositivo (NDC)."""
        p = np.asarray(point, dtype=np.float32)
        p4 = np.append(p, 1.0)
        clip = self.projection_matrix() @ self.view_matrix() @ p4
        if abs(clip[3]) < 1e-9:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return clip[:3] / clip[3]

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Proyecta ``(N, 3)`` puntos 3D a NDC ``(N, 3)`` (vectorizado)."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points debe ser (N,3), se recibió {pts.shape}.")
        vp = (self.projection_matrix() @ self.view_matrix()).astype(np.float64)
        p4 = np.column_stack([pts, np.ones(len(pts))])
        clip = p4 @ vp.T
        w = clip[:, 3:4]
        safe = np.where(np.abs(w) < 1e-9, 1.0, w)
        return clip[:, :3] / safe

    def view_projection_matrix(self) -> np.ndarray:
        """Matriz combinada proyección·vista."""
        return self.projection_matrix() @ self.view_matrix()

    def ray(self, ndc_x: float, ndc_y: float) -> tuple[np.ndarray, np.ndarray]:
        """Rayo (origen, dirección) que pasa por un punto NDC."""
        inv = np.linalg.inv(self.view_projection_matrix())
        p0 = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
        p1 = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float32)
        p0 = p0[:3] / p0[3]
        p1 = p1[:3] / p1[3]
        direction = p1 - p0
        n = np.linalg.norm(direction)
        direction = direction / n if n else np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return p0.astype(np.float32), direction.astype(np.float32)

    @staticmethod
    def ndc_from_pixels(x: float, y: float, width: int, height: int) -> tuple[float, float]:
        """Convierte píxeles (origen arriba-izquierda) a NDC."""
        nx = 2.0 * x / max(width, 1) - 1.0
        ny = 1.0 - 2.0 * y / max(height, 1)
        return nx, ny
