"""Smoke test de la GUI (plataforma offscreen, sin GL)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")  # si está disponible

import numpy as np
import pytest

pyside = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from muni.view.slice_view import SliceView  # noqa: E402
from muni.view.wizard import ReconstructionWizard  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_main_window_constructs(app):
    from muni.view.main_window import MainWindow

    window = MainWindow()
    assert window.viewport is not None
    assert window.meta is None
    window.deleteLater()
    app.processEvents()


def test_wizard_constructs(app):
    wizard = ReconstructionWizard()
    assert wizard.stack.count() == 4
    assert wizard.btn_next is not None
    wizard.deleteLater()
    app.processEvents()


def test_slice_view_qimage(app):
    view = SliceView()
    plane = np.full((30, 40), 200.0, dtype=np.float32)
    plane[10:20, 15:25] = 30.0
    overlay = plane < 100
    view.set_plane(plane, overlay)
    img = view._to_qimage()
    assert not img.isNull()
    assert img.size().width() == 40 and img.size().height() == 30
    view.deleteLater()
    app.processEvents()