# PyInstaller spec para Muni (one-dir multiplataforma).
# Build base SIN torch (el motor U-Net queda como extra opcional).
# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

a = Analysis(
    ["muni/app/main.py"],
    pathex=[],
    binaries=[],
    datas=[("assets", "assets")],
    hiddenimports=["PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PIL._tkinter_finder"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "torch",       # extra de IA opcional (mantener el bundle ligero)
        "torchvision",
        "pytest",
        "matplotlib",
        "pandas",
        "IPython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="muni",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=os.path.abspath(os.path.join("assets", "Muni.ico")),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="muni",
)