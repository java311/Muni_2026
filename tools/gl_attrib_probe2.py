"""Probe 2: exhaustivo de glVertexAttribPointer."""

from __future__ import annotations

import sys
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLFunctions_3_3_Core, QOpenGLVertexArrayObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

f = QSurfaceFormat()
f.setVersion(3, 3)
f.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
QSurfaceFormat.setDefaultFormat(f)

FLOAT = 5126
CASES = [
    ("ints_all", dict(glVertexAttribPointer=(0, 3, FLOAT, 0, 0, 0))),
    ("offset_none", dict(glVertexAttribPointer=(0, 3, FLOAT, 0, 0, None))),
]


class W(QOpenGLWidget):
    def initializeGL(self):
        gl = QOpenGLFunctions_3_3_Core()
        gl.initializeOpenGLFunctions()
        self.vao = QOpenGLVertexArrayObject(self)
        self.vao.create()
        self.vao.bind()
        import numpy as np

        vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.allocate(np.zeros((3, 3), np.float32).tobytes(), 36)
        try:
            gl.glEnableVertexAttribArray(0)
            print("  enableVertexAttribArray OK", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  enable ERR {exc}", flush=True)
        for name, (fn, args) in CASES.items():
            try:
                getattr(gl, fn)(*args)
                print(f"OK  {name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"ERR {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        vbo.release()
        self.vao.release()
        QTimer.singleShot(200, app.quit)

    def paintGL(self):
        pass


app = QApplication(sys.argv)
w = W()
w.resize(120, 90)
w.show()
app.exec()
print("fin", flush=True)