"""Ventana principal de Muni (visor 3D + menús + panel de medición)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from muni import __version__
from muni.app.resources import load_icon
from muni.core.gltf_io import load_model, save_model
from muni.core.meta import ModelMeta
from muni.io.lif_converter import convert_lif_to_png
from muni.trace.swc import write_swc
from muni.view.gl_viewport import GLViewport
from muni.view.wizard import ReconstructionWizard


def _best_icon(*names: str) -> QIcon:
    for name in names:
        icon = load_icon(name)
        if not icon.isNull():
            return icon
    return QIcon()


class MainWindow(QMainWindow):
    """Visor 3D principal."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Muni {__version__}")
        self.setWindowIcon(load_icon("Muni.ico"))
        self.resize(1280, 800)

        self.meta: ModelMeta | None = None
        self.trace_result = None
        self.trace_spacing: tuple[float, float, float] | None = None

        # ------------------------------------------------------------- central
        self.viewport = GLViewport()
        self.viewport.measure_changed.connect(self._on_measure)
        self.setCentralWidget(self.viewport)

        self._build_menus()
        self._build_toolbar()
        self._build_measure_dock()
        self.statusBar().showMessage("Listo — usa Nuevo (Ctrl+N) para reconstruir una pila.")

    # ------------------------------------------------------------------ menús
    def _build_menus(self) -> None:
        mb = self.menuBar()

        menu_file = mb.addMenu("&Archivo")
        act_new = QAction("Nueva reconstrucción…", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._new_from_stack)
        menu_file.addAction(act_new)

        act_open = QAction("Abrir modelo…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_model)
        menu_file.addAction(act_open)

        act_lif = QAction("Convertir .lif a PNG…", self)
        act_lif.triggered.connect(self._convert_lif_to_png)
        menu_file.addAction(act_lif)

        menu_file.addSeparator()

        act_save = QAction("Guardar modelo como…", self)
        act_save.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save.triggered.connect(self._save_model)
        menu_file.addAction(act_save)

        menu_file.addSeparator()
        act_exit = QAction("Salir", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        menu_view = mb.addMenu("&Ver")
        act_axes = QAction("Ejes", self, checkable=True, checked=True)
        act_axes.toggled.connect(lambda v: setattr(self.viewport, "show_axes", v))
        menu_view.addAction(act_axes)

        act_container = QAction("Caja contenedora", self, checkable=True, checked=True)
        act_container.toggled.connect(lambda v: setattr(self.viewport, "show_container", v))
        menu_view.addAction(act_container)

        self._act_skeleton = QAction("Esqueleto dendrítico", self, checkable=True, checked=False)
        self._act_skeleton.toggled.connect(lambda v: setattr(self.viewport, "show_skeleton", v))
        menu_view.addAction(self._act_skeleton)

        sub_mode = menu_view.addMenu("Modo de visualización")
        act_fill = QAction("Relleno", self)
        act_lines = QAction("Líneas", self)
        act_points = QAction("Puntos", self)
        act_fill.setCheckable(True)
        act_lines.setCheckable(True)
        act_points.setCheckable(True)
        act_fill.setChecked(True)
        for mode, act in (("fill", act_fill), ("lines", act_lines), ("points", act_points)):
            act.triggered.connect(
                lambda _checked=False, m=mode, acts=(act_fill, act_lines, act_points): self._set_mode(m, acts)
            )
            sub_mode.addAction(act)

        menu_color = mb.addMenu("&Color")
        act_color_neuron = QAction("Color de la neurona…", self)
        act_color_neuron.triggered.connect(self._pick_neuron_color)
        menu_color.addAction(act_color_neuron)

        menu_help = mb.addMenu("Ay&uda")
        act_about = QAction("Acerca de Muni", self)
        act_about.triggered.connect(self._about)
        menu_help.addAction(act_about)

    def _set_mode(self, mode: str, actions: list[QAction]) -> None:
        for act in actions:
            act.setChecked(act.text().lower() == mode)
        self.viewport.polygon_mode = mode

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Principal")
        tb.setMovable(False)
        act_new = QAction(_best_icon("new_model32.png", "Muni32.gif"), "Nuevo", self)
        act_new.triggered.connect(self._new_from_stack)
        act_open = QAction(_best_icon("archivoabrir32.gif", "file_neuron.png"), "Abrir", self)
        act_open.triggered.connect(self._open_model)
        act_save = QAction(_best_icon("save_model32.gif", "save_model16.gif"), "Guardar", self)
        act_save.triggered.connect(self._save_model)
        act_reset = QAction(_best_icon("refresh32.gif", "refresh24.gif"), "Ver todo", self)
        act_reset.triggered.connect(self.viewport.reset_camera)
        self.chk_measure = QCheckBox("Medir (regla)")
        self.chk_measure.toggled.connect(lambda v: setattr(self.viewport, "measure_mode", v))

        tb.addAction(act_new)
        tb.addAction(act_open)
        tb.addAction(act_save)
        tb.addSeparator()
        tb.addAction(act_reset)
        tb.addWidget(self.chk_measure)

    def _build_measure_dock(self) -> None:
        dock = QDockWidget("Medición", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        self.lbl_distance = QLabel("—")
        self.lbl_distance.setStyleSheet("font-size: 16pt; font-weight: bold; color: #f90;")
        self.lbl_units = QLabel("µm")
        btn_reset = QPushButton("Reiniciar regla")
        btn_reset.clicked.connect(self.viewport.reset_ruler)

        box = QGroupBox("Distancia")
        lay = QVBoxLayout(box)
        lay.addWidget(self.lbl_distance)
        lay.addWidget(self.lbl_units)
        lay.addWidget(btn_reset)

        info = QLabel("Triángulos: —\nVértices: —\nOrigen: —")
        info.setStyleSheet("color: #888;")
        self.lbl_info = info

        outer = QVBoxLayout()
        outer.addWidget(box)
        outer.addWidget(QLabel("Modelo"))
        outer.addWidget(info)
        outer.addStretch(1)

        w = QWidget()
        w.setLayout(outer)
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.measure_info_label = info

    # ----------------------------------------------------------------- accions
    def _new_from_stack(self) -> None:
        wizard = ReconstructionWizard(self)
        if wizard.exec() == ReconstructionWizard.DialogCode.Accepted and wizard.mesh is not None:
            self.meta = wizard.meta
            self.trace_result = wizard.trace_result
            self.trace_spacing = wizard.trace_spacing
            self.viewport.set_mesh(wizard.mesh, units="um")
            if self.trace_result is not None and self.trace_spacing is not None:
                self.viewport.set_skeleton(self.trace_result)
                self._act_skeleton.setChecked(True)
            else:
                self.viewport.clear_skeleton()
                self._act_skeleton.setChecked(False)
            self._update_info()

    def _open_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Abrir modelo", "", "Modelos (*.glb)")
        if not path:
            return
        try:
            mesh, meta = load_model(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Muni — Error", f"No se pudo abrir el modelo:\n{exc}")
            return
        self.meta = meta
        self.viewport.set_mesh(mesh, units="um")
        # Intentar cargar esqueleto SWC acompañante.
        swc_path = Path(path).with_suffix(".swc")
        if swc_path.exists():
            from muni.trace.swc import load_swc
            try:
                trace_result, trace_spacing = load_swc(swc_path)
                self.trace_result = trace_result
                self.trace_spacing = trace_spacing
                self.viewport.set_skeleton(trace_result)
                self._act_skeleton.setChecked(True)
            except (ValueError, OSError):
                self.trace_result = None
                self.trace_spacing = None
                self.viewport.clear_skeleton()
                self._act_skeleton.setChecked(False)
        else:
            self.trace_result = None
            self.trace_spacing = None
            self.viewport.clear_skeleton()
            self._act_skeleton.setChecked(False)
        self._update_info()
        self.statusBar().showMessage(f"Modelo abierto: {Path(path).name}")

    def _convert_lif_to_png(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir archivo Leica", "", "Archivos Leica (*.lif)"
        )
        if not path:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta de destino")
        if not out_dir:
            return

        try:
            count = convert_lif_to_png(path, out_dir)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Muni — Error", f"Error al convertir .lif:\n{exc}")
            return
        QMessageBox.information(
            self, "Muni", f"Conversión completa.\n{count} archivos PNG guardados en:\n{out_dir}"
        )

    def _save_model(self) -> None:
        mesh = self.viewport._mesh
        if mesh is None:
            QMessageBox.information(self, "Muni", "No hay modelo que guardar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar modelo", "modelo.glb", "Modelos (*.glb)")
        if not path:
            return
        meta = self.meta or ModelMeta()
        save_model(path, mesh, meta)
        if self.trace_result is not None:
            swc_path = Path(path).with_suffix(".swc")
            write_swc(swc_path, self.trace_result, spacing=self.trace_spacing)
            self.statusBar().showMessage(f"Guardado: {path} + {swc_path.name}")
        else:
            self.statusBar().showMessage(f"Guardado: {path}")

    def _pick_neuron_color(self) -> None:
        color = QColorDialog.getColor(self.viewport.neuron_color, self, "Color de la neurona")
        if color.isValid():
            self.viewport.neuron_color = color
            self.viewport.update()

    def _on_measure(self, distance: float, units: str) -> None:
        self.lbl_distance.setText(f"{distance:.2f}")
        self.lbl_units.setText(units)

    def _update_info(self) -> None:
        mesh = self.viewport._mesh
        if mesh is None:
            return
        self.lbl_info.setText(
            f"Triángulos: {mesh.face_count:,}\n"
            f"Vértices: {mesh.vertex_count:,}\n"
            f"Extractor: {self.meta.extractor if self.meta else '—'}"
        )

    def _about(self) -> None:
        QMessageBox.about(
            self,
            f"Muni {__version__}",
            "Reconstrucción 3D de neuronas (Golgi-Cox) desde fotografías axiales.\n\n"
            "Algoritmos de segmentación y extracción de superficie escritos desde cero "
            "(NumPy) con interfaz PySide6. Salida en GLTF 2.0.",
        )