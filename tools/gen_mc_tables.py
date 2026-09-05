"""Genera ``muni/reconstruct/mc_tables.py`` a partir de las tablas canónicas.

Las tablas estándar de Marching Cubes (Paul Bourke, publicadas y de dominio
público tras expirar la patente) se extraen del código original ``MarchingCubes.cs``
para garantizar exactitud, sin transcribirlas a mano.

Uso:
    python tools/gen_mc_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT.parent / "Muni-old" / "source" / "Muni" / "MarchingCubes.cs"
OUTPUT = PROJECT_ROOT / "muni" / "reconstruct" / "mc_tables.py"


def extract_edge_table(text: str) -> list[int]:
    m = re.search(r"edgeTable\s*=\s*new\s+int\s*\[\s*\]\s*\{(.*?)\};", text, re.S)
    if not m:
        raise RuntimeError("No se encontró edgeTable en MarchingCubes.cs")
    body = m.group(1)
    return [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body)]


def extract_tri_table(text: str) -> list[list[int]]:
    m = re.search(r"triTable\s*=\s*new\s+int\s*\[256,\s*16\]\s*\{(.*?)\};", text, re.S)
    if not m:
        raise RuntimeError("No se encontró triTable en MarchingCubes.cs")
    body = m.group(1)
    rows = []
    for row in re.findall(r"\{([^{}]*)\}", body):
        values = [int(v) for v in re.findall(r"-?\d+", row)]
        if len(values) != 16:
            raise RuntimeError(f"Fila de triTable con {len(values)} valores, se esperaban 16.")
        rows.append(values)
    if len(rows) != 256:
        raise RuntimeError(f"triTable tiene {len(rows)} filas, se esperaban 256.")
    return rows


def _fmt_hex_table(edge: list[int]) -> str:
    lines = []
    for i in range(0, len(edge), 8):
        chunk = ", ".join(f"0x{v:03x}" for v in edge[i : i + 8])
        lines.append(f"    {chunk},")
    return "\n".join(lines)


def _fmt_tri_table(tri: list[list[int]]) -> str:
    lines = []
    for row in tri:
        lines.append("    (" + ", ".join(str(v) for v in row) + "),")
    return "\n".join(lines)


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    edge = extract_edge_table(text)
    tri = extract_tri_table(text)
    assert len(edge) == 256

    content = (
        "# Archivo generado por tools/gen_mc_tables.py — NO editar a mano.\n"
        "# Tablas canónicas de Marching Cubes (Paul Bourke).\n\n"
        "# edgeTable[i]: máscara de 12 bits con las aristas intersectadas del cubo i.\n"
        f"EDGE_TABLE = (\n{_fmt_hex_table(edge)}\n)\n\n"
        "# triTable[i]: triángulos como tripletas de índices de arista (-1 = fin).\n"
        f"TRI_TABLE = (\n{_fmt_tri_table(tri)}\n)\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Generado {OUTPUT} ({len(edge)} edge, {len(tri)} tri).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
