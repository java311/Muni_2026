"""Visor 2D de cortes (planos) con zoom/pan y dibujo de líneas (calibración).

Los píxeles quedan en coordenadas de imagen nativas; el dibujo de la línea de
dendrita se entrega al asistente mediante ``line_drawn``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget


class SliceView(QWidget):
    """Widget 2D que muestra un plano en gris con zoom, pan y línea de calibración."""

    line_drawn = Signal(tuple)  # (x1, y1, x2, y2) en coordenadas de imagen

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._plane: np.ndarray | None = None  # (Y, X) float32 en 0..255
        self._overlay: np.ndarray | None = None  # máscara bool del umbral
        self._overlay_color: tuple[int, int, int] = (255, 40, 40)  # (R, G, B)
        self._zoom = 1.0
        self._offset = np.array([0.0, 0.0], dtype=np.float64)  # px pantalla
        self._line: tuple[int, int, int, int] | None = None
        self.draw_mode = False
        self._drawing = False
        self._line_start: QPointF | None = None
        self._last: QPointF | None = None
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ datos
    def set_plane(self, plane: np.ndarray, overlay: np.ndarray | None = None) -> None:
        self._plane = np.asarray(plane, dtype=np.float32)
        self._overlay = np.asarray(overlay, dtype=bool) if overlay is not None else None
        self._zoom = 1.0
        self._offset = np.array([0.0, 0.0], dtype=np.float64)
        self.update()

    def set_overlay(self, overlay: np.ndarray | None) -> None:
        self._overlay = np.asarray(overlay, dtype=bool) if overlay is not None else None
        self.update()

    def set_overlay_color(self, r: int, g: int, b: int) -> None:
        self._overlay_color = (r, g, b)

    def set_line(self, line: tuple[int, int, int, int] | None) -> None:
        self._line = line
        self.update()

    def clear(self) -> None:
        self._plane = None
        self._overlay = None
        self._line = None
        self.update()

    # ------------------------------------------------------------ pintado
    def paintEvent(self, event) -> None:  # noqa: N802 (API Qt)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 22, 26))
        if self._plane is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin plano cargado")
            return

        h, w = self._plane.shape
        img = self._to_qimage()
        target = QRectF(self._offset[0], self._offset[1], w * self._zoom, h * self._zoom)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(target, img)

        if self._line is not None:
            x1, y1, x2, y2 = self._line
            pen = QPen(QColor(0, 200, 255), max(2.0, self._zoom * 1.2))
            painter.setPen(pen)
            p1 = QPointF(self._offset[0] + x1 * self._zoom, self._offset[1] + y1 * self._zoom)
            p2 = QPointF(self._offset[0] + x2 * self._zoom, self._offset[1] + y2 * self._zoom)
            painter.drawLine(p1, p2)
            painter.setBrush(QColor(255, 60, 60))
            painter.drawEllipse(p1, 4, 4)
            painter.drawEllipse(p2, 4, 4)

    def _to_qimage(self) -> QImage:
        plane = np.clip(np.rint(self._plane), 0, 255).astype(np.uint8)
        h, w = plane.shape
        if self._overlay is None:
            rgb = np.repeat(plane[:, :, None], 3, axis=2)
        else:
            rgb = np.repeat(plane[:, :, None], 3, axis=2)
            if self._overlay.shape == plane.shape:
                ov = self._overlay
                rgb[ov, 0] = self._overlay_color[0]
                rgb[ov, 1] = self._overlay_color[1]
                rgb[ov, 2] = self._overlay_color[2]
        data = np.ascontiguousarray(rgb)
        img = QImage(data.data, w, h, data.strides[0], QImage.Format.Format_RGB888)
        return img.copy()

    # ------------------------------------------------------------ interacción
    def _image_pos(self, pos: QPointF) -> tuple[float, float]:
        ix = (pos.x() - self._offset[0]) / self._zoom
        iy = (pos.y() - self._offset[1]) / self._zoom
        return ix, iy

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._plane is None:
            return
        factor = 1.15 ** (event.angleDelta().y() / 120.0)
        cursor = event.position()
        img = self._image_pos(cursor)
        self._zoom = float(np.clip(self._zoom * factor, 0.05, 64.0))
        self._offset[0] = cursor.x() - img[0] * self._zoom
        self._offset[1] = cursor.y() - img[1] * self._zoom
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (API Qt)
        self._last = event.position()
        if self.draw_mode and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._line_start = event.position()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (API Qt)
        if self._drawing and self._line_start is not None:
            # Preview de la línea en curso.
            ix1, iy1 = self._image_pos(self._line_start)
            ix2, iy2 = self._image_pos(event.position())
            self._line = (int(round(ix1)), int(round(iy1)), int(round(ix2)), int(round(iy2)))
            self.update()
        elif self._last is not None and event.buttons() & Qt.MouseButton.RightButton:
            delta = event.position() - self._last
            self._offset += np.array([delta.x(), delta.y()])
            self._last = event.position()
            self.update()
        else:
            self._last = event.position()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (API Qt)
        if self._drawing and self._line_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            ix1, iy1 = self._image_pos(self._line_start)
            ix2, iy2 = self._image_pos(event.position())
            if (ix1, iy1) != (ix2, iy2):
                line = (int(round(ix1)), int(round(iy1)), int(round(ix2)), int(round(iy2)))
                self._line = line
                self.line_drawn.emit(line)
        self._line_start = None
        self._drawing = False
        self._last = None
        self.update()

    def fit_to_window(self) -> None:
        if self._plane is None:
            return
        h, w = self._plane.shape
        self._zoom = min(self.width() / max(w, 1), self.height() / max(h, 1))
        self._offset = np.array([0.0, 0.0], dtype=np.float64)
        self.update()