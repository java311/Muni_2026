"""Viewport OpenGL 3.3 para el visor 3D de Muni.

Renderiza un :class:`MeshData` con iluminación sencilla, ejes, caja contenedora
y la regla de medición (dos marcadores arrastrables). Toda la construcción de
buffers/shader es local a este módulo (no se usa ninguna librería 3D externa).
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QVector3D, QWheelEvent
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFunctions_3_3_Core,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from muni.core.meshdata import MeshData
from muni.view.camera import OrbitCamera
from muni.view import glconstants as GL

_GL_DEBUG = os.environ.get("MUNI_GL_DEBUG") == "1"


def _dbg(*args) -> None:
    if _GL_DEBUG:
        print("[gl]", *args, file=sys.stderr, flush=True)


def _err(*args) -> None:
    """Error GL: siempre visible en consola (no depende de MUNI_GL_DEBUG)."""
    print("[gl-error]", *args, file=sys.stderr, flush=True)

_VERT_MESH = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
out vec3 vNormal;
void main() {
    vNormal = aNormal;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_FRAG_MESH = """
#version 330 core
in vec3 vNormal;
uniform vec3 uColor;
uniform vec3 uLightDir;
out vec4 FragColor;
void main() {
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) { n = -n; }
    float diff = max(dot(n, normalize(uLightDir)), 0.0);
    float amb = 0.38;
    vec3 col = uColor * (amb + (1.0 - amb) * diff);
    FragColor = vec4(col, 1.0);
}
"""

_VERT_LINE = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_FRAG_LINE = """
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
"""


class GLViewport(QOpenGLWidget):
    """Widget OpenGL con cámara orbital, malla, ejes, caja y regla."""

    mesh_changed = Signal()
    measure_changed = Signal(float, str)  # (distancia, unidades)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.camera = OrbitCamera()
        self._mesh: MeshData | None = None
        self._meta_units: str = "um"

        self._show_axes = True
        self._show_container = True
        self._polygon_mode: str = "fill"  # fill | lines | points
        self.neuron_color = QColor(0, 128, 192)

        self._measure_mode = False
        self.ruler: list[np.ndarray] = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([25.0, 0.0, 0.0], dtype=np.float32),
        ]
        self._drag_marker: int | None = None

        # Estado del ratón
        self._last_pos = None
        self._mouse_button: Qt.MouseButton | None = None

        # Esqueleto
        self._show_skeleton = False
        self._skeleton_verts: np.ndarray | None = None  # (E*2, 3) world coords
        self._skeleton_colors: list[tuple[float, float, float]] = []

        # Bandera de re-upload de buffers
        self._mesh_dirty = True
        self._line_dirty = True

        # Recursos GL (se crean en initializeGL)
        self._mesh_program: QOpenGLShaderProgram | None = None
        self._line_program: QOpenGLShaderProgram | None = None
        self._mesh_vao: QOpenGLVertexArrayObject | None = None
        self._mesh_vbo_pos: QOpenGLBuffer | None = None
        self._mesh_vbo_nrm: QOpenGLBuffer | None = None
        self._mesh_ibo: QOpenGLBuffer | None = None
        self._mesh_vertex_count = 0
        self._line_vao: QOpenGLVertexArrayObject | None = None
        self._line_vbo: QOpenGLBuffer | None = None
        self._line_segments: list[tuple[int, int, tuple[float, float, float]]] = []
        self._gl = None

    # -------------------------------------------------------------- propiedades
    @property
    def show_axes(self) -> bool:
        return self._show_axes

    @show_axes.setter
    def show_axes(self, value: bool) -> None:
        self._show_axes = bool(value)
        self._line_dirty = True
        self.update()

    @property
    def show_container(self) -> bool:
        return self._show_container

    @show_container.setter
    def show_container(self, value: bool) -> None:
        self._show_container = bool(value)
        self._line_dirty = True
        self.update()

    @property
    def polygon_mode(self) -> str:
        return self._polygon_mode

    @polygon_mode.setter
    def polygon_mode(self, value: str) -> None:
        if value not in ("fill", "lines", "points"):
            raise ValueError(f"Modo de polígono desconocido: {value!r}")
        self._polygon_mode = value
        self.update()

    @property
    def measure_mode(self) -> bool:
        return self._measure_mode

    @measure_mode.setter
    def measure_mode(self, value: bool) -> None:
        self._measure_mode = bool(value)
        self._line_dirty = True
        self.update()

    # --------------------------------------------------------- esqueleto
    @property
    def show_skeleton(self) -> bool:
        return self._show_skeleton

    @show_skeleton.setter
    def show_skeleton(self, value: bool) -> None:
        self._show_skeleton = bool(value)
        self._line_dirty = True
        self.update()

    def set_skeleton(self, coords: np.ndarray, parents: np.ndarray, spacing: tuple[float, float, float]) -> None:
        """Carga esqueleto para renderizado.

        Parameters
        ----------
        coords:
            ``(N, 3)`` coordenadas de voxel (z, y, x).
        parents:
            ``(N,)`` índice del padre (-1 para raíz).
        spacing:
            ``(dz, dy, dx)`` en micras.
        """
        if coords is None or len(coords) == 0:
            self._skeleton_verts = None
            self._skeleton_colors = []
            self.update()
            return

        # Pequeño offset para que el esqueleto quede ligeramente por encima de la
        # superficie de la neurona (evita z-fighting visual).
        offset = 0.3

        edges = []
        for i in range(len(parents)):
            p = int(parents[i])
            if p >= 0:
                # Convención del mesh (marching cubes): [Z*sz, Y*sy, X*sx].
                v0 = np.array([
                    coords[i, 0] * spacing[0] + offset,
                    coords[i, 1] * spacing[1],
                    coords[i, 2] * spacing[2],
                ], dtype=np.float32)
                v1 = np.array([
                    coords[p, 0] * spacing[0] + offset,
                    coords[p, 1] * spacing[1],
                    coords[p, 2] * spacing[2],
                ], dtype=np.float32)
                edges.append((v0, v1))

        if edges:
            self._skeleton_verts = np.concatenate(
                [np.array([a, b], np.float32) for a, b in edges], axis=0
            )
            self._skeleton_colors = [(1.0, 0.3, 0.3)] * len(edges)
        else:
            self._skeleton_verts = None
            self._skeleton_colors = []
        self._line_dirty = True
        self.update()

    def clear_skeleton(self) -> None:
        self._skeleton_verts = None
        self._skeleton_colors = []
        self._line_dirty = True
        self.update()

    # ------------------------------------------------------------------ API
    def set_mesh(self, mesh: MeshData, units: str = "um") -> None:
        self._mesh = mesh
        self._meta_units = units
        self._mesh_dirty = True
        self._line_dirty = True
        if not mesh.is_empty:
            lo, hi = mesh.bounds()
            self.camera.fit(lo, hi)
            self.ruler = [lo.copy(), hi.copy()]
        self.mesh_changed.emit()
        self.emit_measure()
        self.update()

    def clear_mesh(self) -> None:
        self._mesh = None
        self._mesh_dirty = True
        self._line_dirty = True
        self.update()

    def reset_camera(self) -> None:
        if self._mesh is not None and not self._mesh.is_empty:
            lo, hi = self._mesh.bounds()
            self.camera.fit(lo, hi)
        else:
            self.camera = OrbitCamera()
        self.update()

    def measure_distance(self) -> float:
        return float(np.linalg.norm(self.ruler[1] - self.ruler[0]))

    def reset_ruler(self) -> None:
        """Coloca la regla sobre los extremos de la malla (o el origen)."""
        if self._mesh is not None and not self._mesh.is_empty:
            lo, hi = self._mesh.bounds()
            self.ruler = [lo.copy(), hi.copy()]
        else:
            self.ruler = [
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([25.0, 0.0, 0.0], dtype=np.float32),
            ]
        self._line_dirty = True
        self.emit_measure()
        self.update()

    # ------------------------------------------------------- OpenGL lifecycle
    def initializeGL(self) -> None:
        try:
            self._initialize_gl()
        except Exception:  # noqa: BLE001 - nunca dejar el viewport roto en silencio
            _err("initializeGL falló:\n", traceback.format_exc())

    def _initialize_gl(self) -> None:
        # PySide6 >= 6.x: QOpenGLContext.versionFunctions() ya no existe.
        # Se usa QOpenGLFunctions_3_3_Core con initializeOpenGLFunctions()
        # (el contexto está activo dentro de initializeGL).
        if self._gl is None:
            self._gl = QOpenGLFunctions_3_3_Core()
            self._gl.initializeOpenGLFunctions()
        gl = self._gl
        gl.glEnable(GL.GL_DEPTH_TEST)
        _dbg("initializeGL ok, renderer disponible (QOpenGLFunctions_3_3_Core)")

        self._mesh_program = self._build_program(_VERT_MESH, _FRAG_MESH)
        self._line_program = self._build_program(_VERT_LINE, _FRAG_LINE)

        # VAO/VBO de la malla (posiciones, normales, índices)
        self._mesh_vao = QOpenGLVertexArrayObject(self)
        self._mesh_vao.create()
        self._mesh_vbo_pos = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._mesh_vbo_nrm = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._mesh_ibo = QOpenGLBuffer(QOpenGLBuffer.Type.IndexBuffer)

        # VAO/VBO de líneas (ejes, caja, regla)
        self._line_vao = QOpenGLVertexArrayObject(self)
        self._line_vao.create()
        self._line_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._line_segments = []

        self.upload_mesh()
        self.update()

    def _build_program(self, vsrc: str, fsrc: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram(self)
        if not prog.addShaderFromSourceCode(QOpenGLShader.Vertex, vsrc):
            raise RuntimeError(f"Error compilando vertex shader: {prog.log()}")
        if not prog.addShaderFromSourceCode(QOpenGLShader.Fragment, fsrc):
            raise RuntimeError(f"Error compilando fragment shader: {prog.log()}")
        if not prog.link():
            raise RuntimeError(f"Error enlazando shader: {prog.log()}")
        return prog

    def _mesh_ready(self) -> bool:
        return (
            self._mesh is not None
            and not self._mesh.is_empty
            and self._mesh_vao is not None
        )

    def upload_mesh(self) -> None:
        gl = self._gl
        if gl is None or self._mesh_vao is None:
            self._mesh_dirty = True
            return
        self._mesh_vertex_count = 0
        if self._mesh is None or self._mesh.is_empty:
            self._mesh_dirty = False
            return

        self._mesh_vao.bind()

        # Se expanden los índices (sopa de triángulos) para dibujar con
        # glDrawArrays: PySide6 6.11 no enlaza bien glDrawElements/glVertexAttribPointer.
        verts = self._mesh.vertices[self._mesh.faces].reshape(-1, 3)  # (F*3, 3)
        norms = self._mesh.normals[self._mesh.faces].reshape(-1, 3)

        self._mesh_vbo_pos.create()
        self._mesh_vbo_pos.bind()
        self._mesh_vbo_pos.allocate(verts.tobytes(), verts.nbytes)
        self._mesh_program.enableAttributeArray(0)
        self._mesh_program.setAttributeBuffer(0, GL.GL_FLOAT, 0, 3, 0)
        self._mesh_vbo_pos.release()

        self._mesh_vbo_nrm.create()
        self._mesh_vbo_nrm.bind()
        self._mesh_vbo_nrm.allocate(norms.tobytes(), norms.nbytes)
        self._mesh_program.enableAttributeArray(1)
        self._mesh_program.setAttributeBuffer(1, GL.GL_FLOAT, 0, 3, 0)
        self._mesh_vbo_nrm.release()

        self._mesh_vao.release()
        self._mesh_vertex_count = 3 * self._mesh.face_count
        self._mesh_dirty = False

    def _rebuild_lines(self) -> None:
        """Reconstruye el buffer de líneas (ejes, caja, regla, marcadores)."""
        if self._gl is None or self._line_vao is None:
            return
        segments: list[tuple[np.ndarray, tuple[float, float, float]]] = []

        if self.show_axes:
            size = self._line_extent()
            segments += [
                (np.array([[0, 0, 0], [size, 0, 0]], np.float32), (1.0, 0.0, 0.0)),
                (np.array([[0, 0, 0], [0, size, 0]], np.float32), (0.0, 1.0, 0.0)),
                (np.array([[0, 0, 0], [0, 0, size]], np.float32), (0.0, 0.6, 1.0)),
            ]

        if self.show_container and self._mesh is not None and not self._mesh.is_empty:
            lo, hi = self._mesh.bounds()
            c = (0.9, 0.7, 0.2)
            for edge in _box_edges(lo, hi):
                segments.append((np.asarray(edge, np.float32), c))

        # Regla: línea entre marcadores + cruces en los marcadores.
        if self.measure_mode:
            r0, r1 = self.ruler
            segments.append((np.array([r0, r1], np.float32), (1.0, 0.6, 0.0)))
            size = self._line_extent() * 0.02
            for p, col in ((r0, (0.0, 1.0, 0.0)), (r1, (1.0, 0.3, 0.3))):
                for axis in range(3):
                    a = p.copy()
                    b = p.copy()
                    a[axis] -= size
                    b[axis] += size
                    segments.append((np.array([a, b], np.float32), col))

        # Esqueleto: aristas del árbol dendrítico.
        if self._show_skeleton and self._skeleton_verts is not None and len(self._skeleton_verts) > 0:
            segments.append((self._skeleton_verts, (1.0, 0.3, 0.3)))

        # Concatenar en un solo VBO, recordando (offset, count, color).
        if segments:
            all_verts = np.concatenate([seg[0] for seg in segments], axis=0).astype(np.float32)
            if not self._line_vbo.isCreated():
                self._line_vbo.create()
            self._line_vao.bind()
            self._line_vbo.bind()
            self._line_vbo.allocate(all_verts.tobytes(), all_verts.nbytes)
            # Atributo configurado con la API de QOpenGLShaderProgram (segura).
            self._line_program.enableAttributeArray(0)
            self._line_program.setAttributeBuffer(0, GL.GL_FLOAT, 0, 3, 0)
            self._line_vbo.release()
            self._line_vao.release()
            self._line_segments = []
            offset = 0
            for verts, color in segments:
                count = verts.shape[0]
                self._line_segments.append((offset, count, color))
                offset += count
        else:
            self._line_segments = []

    def _line_extent(self) -> float:
        if self._mesh is not None and not self._mesh.is_empty:
            lo, hi = self._mesh.bounds()
            return float(np.max(hi - lo)) or 1.0
        return self.camera.distance

    def paintGL(self) -> None:
        try:
            self._paint_gl()
        except Exception:  # noqa: BLE001 - el error se loguea, no aborta el frame en silencio
            _err("paintGL falló:\n", traceback.format_exc())

    def _paint_gl(self) -> None:
        gl = self._gl
        if gl is None:
            return
        _dbg("paintGL", self.width(), "x", self.height(), "mesh_dirty", self._mesh_dirty)
        gl.glClearColor(0.1, 0.12, 0.14, 1.0)
        gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if self._mesh_dirty:
            self.upload_mesh()
        if self._line_dirty:
            self._rebuild_lines()
            self._line_dirty = False

        self.camera.set_aspect(self.width(), self.height())
        mvp = _qmatrix4x4(self.camera.view_projection_matrix())

        if self._mesh_ready():
            mode = {"fill": GL.GL_FILL, "lines": GL.GL_LINE, "points": GL.GL_POINT}[
                self.polygon_mode
            ]
            gl.glPolygonMode(GL.GL_FRONT_AND_BACK, mode)
            self._mesh_program.bind()
            self._mesh_program.setUniformValue("uMVP", mvp)
            col = self.neuron_color
            # QVector3D en lugar de QColor: QColor no enlaza bien con vec3.
            self._mesh_program.setUniformValue(
                "uColor", QVector3D(col.redF(), col.greenF(), col.blueF())
            )
            self._mesh_program.setUniformValue("uLightDir", QVector3D(-0.4, -0.8, -0.4))
            self._mesh_vao.bind()
            _dbg("dibujando", self._mesh_vertex_count, "vértices (triángulos expandidos)")
            gl.glDrawArrays(GL.GL_TRIANGLES, 0, self._mesh_vertex_count)
            self._mesh_vao.release()
            self._mesh_program.release()
            gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # Líneas
        if self._line_vbo is not None and self._line_segments:
            gl.glDisable(GL.GL_DEPTH_TEST)
            self._line_program.bind()
            self._line_program.setUniformValue("uMVP", mvp)
            self._line_vao.bind()
            self._line_vbo.bind()
            for offset, count, color in self._line_segments:
                self._line_program.setUniformValue("uColor", QVector3D(*color))
                gl.glDrawArrays(GL.GL_LINES, offset, count)
            self._line_vbo.release()
            self._line_vao.release()
            self._line_program.release()
            gl.glEnable(GL.GL_DEPTH_TEST)

    # ------------------------------------------------------------- interacción
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_pos = event.position()
        self._mouse_button = event.button()
        if self.measure_mode and event.button() == Qt.MouseButton.LeftButton:
            marker = self._pick_marker(event.position().x(), event.position().y())
            if marker is not None:
                self._drag_marker = marker
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_pos is None:
            self._last_pos = event.position()
            return
        dx = event.position().x() - self._last_pos.x()
        dy = event.position().y() - self._last_pos.y()

        if self._drag_marker is not None:
            self._drag_marker_to(event.position().x(), event.position().y(), self._drag_marker)
            self._line_dirty = True
            self.emit_measure()
        elif self._mouse_button == Qt.MouseButton.LeftButton:
            # Rotación orbital
            self.camera.rotate(dx * 0.4, dy * -0.4)
        elif self._mouse_button in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self.camera.pan(dx, dy, self.height())

        self._last_pos = event.position()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_marker = None
        self._mouse_button = None
        self._last_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        self.camera.zoom(1.0 - delta / 1200.0)
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_F:
            self.reset_camera()

    # ---------------------------------------------------------------- helpers
    def _pick_marker(self, x: float, y: float) -> int | None:
        best: tuple[float, int] | None = None
        for i, p in enumerate(self.ruler):
            ndc = self.camera.project(p)
            sx = (ndc[0] + 1.0) * 0.5 * self.width()
            sy = (1.0 - ndc[1]) * 0.5 * self.height()
            dist = float(np.hypot(sx - x, sy - y))
            if best is None or dist < best[0]:
                best = (dist, i)
        if best is not None and best[0] < 15.0:
            return best[1]
        return None

    def _drag_marker_to(self, x: float, y: float, marker: int) -> None:
        """Mueve el marcador sobre el plano perpendicular a la vista en su profundidad."""
        ndc_x, ndc_y = self.camera.ndc_from_pixels(x, y, self.width(), self.height())
        origin, direction = self.camera.ray(ndc_x, ndc_y)
        point = self.ruler[marker]
        forward = self.camera.target - self.camera.eye()
        n = float(np.linalg.norm(forward))
        if n == 0:
            return
        forward = forward / n
        denom = float(np.dot(forward, direction))
        if abs(denom) < 1e-6:
            return
        t = float(np.dot(forward, point - origin)) / denom
        self.ruler[marker] = (origin + t * direction).astype(np.float32)

    def emit_measure(self) -> None:
        self.measure_changed.emit(self.measure_distance(), self._meta_units)


# ------------------------------------------------------------------ utilidades
def _qmatrix4x4(m: np.ndarray):
    from PySide6.QtGui import QMatrix4x4

    flat = m.ravel().tolist()
    return QMatrix4x4(*flat)


def _box_edges(lo: np.ndarray, hi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Las 12 aristas de una caja (para el contenedor)."""
    lo = np.asarray(lo, np.float32)
    hi = np.asarray(hi, np.float32)
    corners = np.array(
        [
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [(corners[a], corners[b]) for a, b in edges]