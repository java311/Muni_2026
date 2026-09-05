"""Verificación manual del visor GL: crea una malla sintética y pinta una ventana.

Uso:
    python tools/gl_smoke.py            # plataforma por defecto
    QT_QPA_PLATFORM=offscreen python tools/gl_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def main() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # Malla sintética: esfera por Marching Cubes sobre un campo de probabilidad.
    from muni.reconstruct.marching_cubes import MarchingCubes
    from muni.view.main_window import MainWindow

    z, y, x = np.mgrid[0:32, 0:32, 0:32].astype(np.float32)
    d = np.sqrt((z - 16) ** 2 + (y - 16) ** 2 + (x - 16) ** 2)
    field = (1.0 / (1.0 + np.exp((d - 10.0) / 1.0))).astype(np.float32)
    mc = MarchingCubes(inside_high=True)
    mesh = mc.extract(field, isovalue=0.5, spacing=(1.0, 1.0, 1.0)).mesh
    print(f"Malla: {mesh.vertex_count} vértices, {mesh.face_count} triángulos")

    win = MainWindow()
    win.viewport.set_mesh(mesh)
    win.viewport.measure_mode = True
    win.show()
    print("Ventana mostrada, ejecutando bucle de eventos...", flush=True)

    QTimer.singleShot(1500, app.quit)
    code = app.exec()
    print(f"app.exec() devolvió {code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())