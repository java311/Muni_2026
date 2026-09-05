"""Pruebas del reporte de errores en consola (worker + hooks)."""

import sys

import pytest

from muni.app.main import install_error_reporting


def test_worker_error_emit_traceback():
    from muni.view.worker import Worker

    def boom(progress):
        raise RuntimeError("fallo_deliberado")

    errors: list[str] = []
    worker = Worker(boom)
    worker.error.connect(errors.append)
    worker.run()
    assert len(errors) == 1
    assert "Traceback" in errors[0]
    assert "fallo_deliberado" in errors[0]


def test_worker_success_emits_finished():
    from muni.view.worker import Worker

    done: list = []
    worker = Worker(lambda progress: 42)
    worker.finished.connect(done.append)
    worker.run()
    assert done == [42]


def test_excepthook_prints_to_stderr(capsys):
    install_error_reporting()
    assert sys.excepthook is not None
    try:
        raise ValueError("error_visible")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    err = capsys.readouterr().err
    assert "muni-error" in err
    assert "ValueError: error_visible" in err
    assert "Traceback" in err