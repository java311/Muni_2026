"""Punto de entrada de la aplicación Muni (interfaz PySide6)."""

from __future__ import annotations

import datetime
import os
import sys
import threading
import traceback


def install_error_reporting() -> None:
    """Hace visibles en la consola (stderr) todos los errores posibles.

    - Excepciones de Python no capturadas (sys.excepthook y threading.excepthook).
    - Mensajes de Qt (advertencias de OpenGL, plugins, etc.), que por defecto
      en Windows van a OutputDebugString y NO se ven en consola.
    - Además se escriben en un log persistente (%LOCALAPPDATA%/muni/muni.log).

    Llamar antes de crear QApplication.
    """
    import warnings

    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    def _log_path() -> str:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        folder = os.path.join(base, "muni")
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            pass
        return os.path.join(folder, "muni.log")

    def _append_log(text: str) -> None:
        try:
            with open(_log_path(), "a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            pass

    def _python_excepthook(etype, value, tb) -> None:
        text = "".join(traceback.format_exception(etype, value, tb))
        print(f"[muni-error]{os.linesep}{text}", file=sys.stderr, flush=True)
        _append_log(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] ERROR{os.linesep}{text}{os.linesep}")

    sys.excepthook = _python_excepthook
    threading.excepthook = lambda args: _python_excepthook(*args.exc_info)

    _SEV = {
        QtMsgType.QtDebugMsg: "DEBUG",
        QtMsgType.QtInfoMsg: "INFO",
        QtMsgType.QtWarningMsg: "WARNING",
        QtMsgType.QtCriticalMsg: "CRITICAL",
        QtMsgType.QtFatalMsg: "FATAL",
    }

    def _qt_message_handler(mode, _context, message) -> None:
        sev = _SEV.get(mode, "?")
        line = f"[qt-{sev}] {message}"
        print(line, file=sys.stderr, flush=True)
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            _append_log(
                f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] QT-{sev} {message}{os.linesep}"
            )

    qInstallMessageHandler(_qt_message_handler)
    # Conservar referencia: evita que el handler se recoja como basura.
    globals()["_muni_qt_handler"] = _qt_message_handler
    globals()["_muni_py_hook"] = _python_excepthook

    warnings.filterwarnings("default")


def main(argv: list[str] | None = None) -> int:
    """Arranca la aplicación. Debe configurarse el formato OpenGL antes de QApplication."""
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    install_error_reporting()

    argv = sys.argv if argv is None else argv
    smoke = "--smoke" in argv

    # Contexto OpenGL 3.3 core (los shaders requieren GLSL 330).
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(argv)
    app.setApplicationName("Muni")
    app.setOrganizationName("Muni")

    from muni.view.main_window import MainWindow

    window = MainWindow()

    if smoke:  # verificación: malla sintética y cierre automático
        from PySide6.QtCore import QTimer

        import numpy as np

        from muni.reconstruct.marching_cubes import MarchingCubes

        z, y, x = np.mgrid[0:24, 0:24, 0:24].astype(np.float32)
        d = np.sqrt((z - 12) ** 2 + (y - 12) ** 2 + (x - 12) ** 2)
        field = (1.0 / (1.0 + np.exp((d - 8.0) / 1.0))).astype(np.float32)
        mesh = MarchingCubes(inside_high=True).extract(field, 0.5).mesh
        window.viewport.set_mesh(mesh)
        QTimer.singleShot(1500, lambda: (window.close(), app.quit()))

    window.show()
    code = app.exec()

    # Workaround: el GC del intérprete al cerrar puede corromper el teardown de
    # Qt/OpenGL en laptops con GPU híbrida (NVIDIA/AMD), causando un acceso
    # ilegal. Con os._exit el proceso termina sin ejecutar ese GC final.
    try:
        os._exit(code)  # noqa: PLR1722 - intencionado
    except BaseException:  # noqa: BLE001 - si algo impide _exit, se devuelve normal
        return code


if __name__ == "__main__":
    raise SystemExit(main())