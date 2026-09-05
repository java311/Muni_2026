"""Ejecución de tareas largas en un hilo Qt (QThread) con progreso y cancelación.

No se usan ``BackgroundWorker``: es el equivalente moderno con señales Qt.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

ProgressCallback = Callable[[int, str], None]


class Worker(QObject):
    """Ejecuta ``fn(progress)`` en un hilo; emite ``finished``/``error``/``progress``."""

    finished = Signal(object)  # resultado (puede ser None)
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, fn: Callable[[ProgressCallback], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._fn = fn

    @Slot()
    def run(self) -> None:
        def report(percent: int, message: str) -> None:
            self.progress.emit(int(percent), message)

        try:
            result = self._fn(report)
        except Exception:  # noqa: BLE001 - se reporta al hilo principal (traceback completo)
            self.error.emit(traceback.format_exc())
            return
        self.finished.emit(result)


def run_in_thread(
    fn: Callable[[ProgressCallback], Any],
    on_finished: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_progress: Callable[[int, str], None] | None = None,
) -> QThread:
    """Arranca ``fn`` en un hilo y conecta las señales. Devuelve el ``QThread``.

    Es importante conservar una referencia fuerte al ``Worker`` (``thread.worker``):
    si solo se guarda el ``QThread``, el recolector de basura puede destruir el
    worker antes de que el hilo procese ``started`` y la tarea jamás se ejecuta
    (botón sin efecto, sin progreso ni error).
    """
    thread = QThread()
    worker = Worker(fn)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    if on_progress is not None:
        worker.progress.connect(on_progress)

    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.error.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    # Referencia fuerte del worker mientras el hilo esté vivo.
    thread.worker = worker  # type: ignore[attr-defined]

    thread.start()
    return thread