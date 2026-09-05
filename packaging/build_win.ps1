# Compila la aplicación con PyInstaller (Windows).
# Uso: powershell -ExecutionPolicy Bypass -File packaging\build_win.ps1 [-Clean]

param([switch]$Clean)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

& .\.venv\Scripts\python.exe -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) { exit 1 }

if ($Clean) {
    Remove-Item dist\muni, build -Recurse -Force -ErrorAction SilentlyContinue
}

& .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean muni.spec
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Listo: dist\muni\muni.exe"