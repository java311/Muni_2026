"""Constantes OpenGL usadas por el visor.

PySide6 6.x no expone las constantes ``GL_*`` como atributos de
``QOpenGLFunctions_3_3_Core``; se definen aquí con sus valores canónicos.
"""

from __future__ import annotations

# Enables/bitfields
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_COLOR_BUFFER_BIT = 0x00004000

# Enables
GL_DEPTH_TEST = 0x0B71
GL_FRONT_AND_BACK = 0x0408

# Primitivas
GL_POINTS = 0x0000
GL_LINES = 0x0001
GL_TRIANGLES = 0x0004

# Tipos
GL_FLOAT = 0x1406
GL_UNSIGNED_INT = 0x1405

# PolygonMode
GL_POINT = 0x1B00
GL_LINE = 0x1B01
GL_FILL = 0x1B02