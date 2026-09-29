# Muni 2 — Reconstrucción 3D de neuronas

Reescritura en **Python + PySide6** del software **Muni** (originalmente C# / WinForms /
Tao.OpenGL), que reconstruye en 3D neuronas teñidas con el método de **Golgi-Cox** a partir
de una pila de fotografías axiales tomadas con microscopio óptico, y permite medir distancias
sobre el modelo resultante.

Los algoritmos de reconstrucción están **escritos desde cero** (sin librerías de
reconstrucción ya hechas); solo se usa NumPy para el cálculo vectorial, PySide6/Qt para la
interfaz y OpenGL para la visualización.

## Estado

En desarrollo. Fases completadas:

- **F0** — Estructura del repositorio, assets heredados y datos de prueba extraídos.
- **F1** — Núcleo de datos: volumen, malla, metadatos, cargador de pilas y exportador GLTF.
- **F2** — Preprocesado (difusión anisotrópica) y segmentación clásica (Otsu + morfología).
- **F3** — Extracción de superficie: **Marching Cubes** y **Dual Contouring** desde cero + suavizado.
- **F4** — Calibración: umbral automático y resolución Z por dendrita.
- **F5** — Interfaz PySide6: asistente de 4 pasos, visor 3D OpenGL, regla de medición e iconos heredados.
- **F6** — Validación con datos reales (genera GLB desde los paquetes; `tools/validate_real_stack.py`).
- **F7** — Segmentación U-Net opcional (PyTorch, carga diferida) con entrenamiento por pseudo-etiquetas
  del motor clásico (`tools/train_unet.py`).
- **F8** — Empaquetado con PyInstaller (`muni.spec`, `packaging/build_win.ps1`).

## Requisitos

- Python 3.10+
- `pip install -e ".[dev]"` (desarrollo) o `pip install -e .`
- Extra opcional de IA: `pip install -e ".[ai]"` (solo para el motor U-Net)

## Uso

```bash
python -m muni.app.main       # aplicación GUI
python tools/validate_real_stack.py --downsample 2   # pipeline sin GUI sobre un paquete
python tools/train_unet.py --epochs 8                # entrena el U-Net opcional
```

## Empaquetado (PyInstaller)

```bash
# Windows
powershell -ExecutionPolicy Bypass -File packaging\build_win.ps1
# Resultado: dist\muni\muni.exe

# macOS / Linux (tras instalar pyinstaller en el venv)
python -m PyInstaller --noconfirm --clean muni.spec
```

El bundle base excluye `torch` para mantenerse ligero; el motor U-Net solo está
disponible en ejecución desde el intérprete con el extra `[ai]` instalado.

## Pruebas

```bash
python -m pytest -q
```

## Unidades y escala

La malla se construye en **micras** aplicando la calibración:
- `spacing_xy_um` (µm/píxel) y `spacing_z_um` (µm/plano) se derivan del diámetro
  de la dendrita marcada en el paso de calibración del asistente.
- Los vértices se escalan por esos espaciados, por lo que la malla y la regla de
  medición quedan en unidades físicas reales.

Si la calibración Z falla (no se detecta la dendrita a lo largo de Z), el
asistente avisa y se usa un espaciado por defecto de 1.0, con lo que la malla
queda en unidades de píxel (una pila de pocos planos se ve "achatada"). Para un
resultado correcto, marca una dendrita con una línea corta que la cruce e
indica su diámetro en micras.

## Errores en consola

Todos los errores se muestran en la **consola (stderr)** con prefijos:
`[muni-error]` (excepciones de Python con traceback completo), `[qt-*]`
(advertencias/errores de Qt, incluidos los de OpenGL) y `[gl-error]` (errores
de renderizado). Además se registran en `%LOCALAPPDATA%\muni\muni.log`
(Windows) o `~/.muni/muni.log` (Linux/macOS).

Sugerencia para ver todos los detalles al ejecutar la app desde consola:

```bash
python -m muni.app.main 2>&1            # Windows (PowerShell/cmd)
python -m muni.app.main                # Linux/macOS (stderr visible)
```

## Notas del visor OpenGL

En laptops con GPU híbrida (NVIDIA + AMD) hay un fallo conocido de
PySide6/QOpenGLWidget al salir de la aplicación (acceso violado en el GC del
intérprete). `muni/app/main.py` usa `os._exit` tras el bucle de eventos para
evitarlo. El resto del funcionamiento (ventana, renderizado, cierre) es normal.

## Papers
Li, R., Kudryashev, M. & Yakimovich, A. "A weak-labelling and deep learning approach for in-focus object segmentation in 3D widefield microscopy." Scientific Reports 13, 12275 (2023).  (revisar paper)

## Licencia

GPL-3.0-or-later. El proyecto original Muni se distribuye bajo GNU/GPL.

## TODO TASKS (for user's reference not for the model reference)
-- Unet check if it is used or not. If not delete it. 
-- Update documentation (uml diagrams of connection between classes may be good)
-- Only make the 3d model where there is skeleton. 
-- change skeleton from spheres to pipes
-- update documentation (for the internet)
-- spines automatic count
-- Fiji export file
-- Understand every property and value in the wizard
-- 
