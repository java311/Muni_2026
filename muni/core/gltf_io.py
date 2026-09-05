"""Exportación/importación de mallas en GLTF 2.0 (GLB) — desde cero.

GLB = cabecera (12 bytes) + chunk JSON + chunk BIN. Este módulo no depende de
ninguna librería de GLTF; construye el JSON y el buffer binario manualmente y
usa ``struct``/``numpy`` para serializar.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from muni.core.meshdata import MeshData
from muni.core.meta import ModelMeta

GLB_MAGIC = 0x46546C67  # "glTF"
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A  # "JSON"
CHUNK_BIN = 0x004E4942  # "BIN\0"

FLOAT = 5126
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
TRIANGLES = 4


def _pad(data: bytes, byte: int = 0x00) -> bytes:
    """Rellena a un múltiplo de 4 bytes."""
    rem = len(data) % 4
    if rem:
        data += bytes([byte]) * (4 - rem)
    return data


def _build_binary(mesh: MeshData) -> bytes:
    """Serializa POSITION (float32) + NORMAL (float32) + indices (uint32)."""
    positions = np.ascontiguousarray(mesh.vertices, dtype="<f4")
    normals = np.ascontiguousarray(mesh.normals, dtype="<f4")
    indices = np.ascontiguousarray(mesh.faces, dtype="<u4")
    return positions.tobytes() + normals.tobytes() + indices.tobytes()


def _build_json(mesh: MeshData) -> dict:
    n = mesh.vertex_count
    m = mesh.face_count

    pos_len = n * 3 * 4
    nrm_len = n * 3 * 4
    idx_len = m * 3 * 4

    pos_min = mesh.vertices.min(axis=0).tolist()
    pos_max = mesh.vertices.max(axis=0).tolist()

    # El material se guarda aunque GLTF no requiera texturas; así los visores
    # muestran el color de la neurona con doble cara activada.
    base_color = [0.0, 0.502, 0.753, 1.0]  # (0,128,192)/255

    return {
        "asset": {"version": "2.0", "generator": "muni"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": mesh.name or "neuron"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                        "mode": TRIANGLES,
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "neuron",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": base_color,
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
            }
        ],
        "buffers": [{"byteLength": pos_len + nrm_len + idx_len}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": nrm_len, "target": ARRAY_BUFFER},
            {
                "buffer": 0,
                "byteOffset": pos_len + nrm_len,
                "byteLength": idx_len,
                "target": ELEMENT_ARRAY_BUFFER,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": FLOAT,
                "count": n,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            },
            {
                "bufferView": 1,
                "componentType": FLOAT,
                "count": n,
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": UNSIGNED_INT,
                "count": m * 3,
                "type": "SCALAR",
            },
        ],
    }


def write_glb(mesh: MeshData, path: str | Path) -> Path:
    """Escribe ``mesh`` como archivo GLB."""
    path = Path(path)
    bin_data = _build_binary(mesh)
    json_bytes = json.dumps(_build_json(mesh), separators=(",", ":")).encode("utf-8")
    json_bytes = _pad(json_bytes, 0x20)
    bin_chunk = _pad(bin_data, 0x00)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_chunk)
    header = struct.pack("<III", GLB_MAGIC, GLB_VERSION, total)
    json_chunk = struct.pack("<II", len(json_bytes), CHUNK_JSON) + json_bytes
    bin_header = struct.pack("<II", len(bin_chunk), CHUNK_BIN) + bin_chunk

    path.write_bytes(header + json_chunk + bin_header)
    return path


def read_glb(path: str | Path) -> MeshData:
    """Lee un GLB (POSITION + NORMAL + indices) y devuelve un :class:`MeshData`."""
    path = Path(path)
    data = path.read_bytes()

    if len(data) < 12:
        raise ValueError("GLB truncado (cabecera incompleta).")
    magic, version, _total = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("No es un archivo GLB (magic incorrecto).")
    if version != 2:
        raise ValueError(f"Versión GLB no soportada: {version}.")

    offset = 12
    json_blob: Optional[bytes] = None
    bin_blob: Optional[bytes] = None
    while offset + 8 <= len(data):
        (length, ctype) = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + length]
        offset += length
        if ctype == CHUNK_JSON:
            json_blob = chunk
        elif ctype == CHUNK_BIN:
            bin_blob = chunk
        else:
            continue  # chunk desconocido; se ignora

    if json_blob is None or bin_blob is None:
        raise ValueError("GLB sin chunk JSON o BIN.")

    doc = json.loads(json_blob.decode("utf-8"))
    primitive = doc["meshes"][0]["primitives"][0]
    attrs = primitive["attributes"]
    pos_acc = doc["accessors"][attrs["POSITION"]]
    nrm_acc = doc["accessors"][attrs.get("NORMAL", attrs["POSITION"])]
    idx_acc = doc["accessors"][primitive["indices"]]

    def accessor_array(acc: dict) -> np.ndarray:
        view = doc["bufferViews"][acc["bufferView"]]
        off = view.get("byteOffset", 0)
        raw = bin_blob[off : off + view["byteLength"]]
        if acc["componentType"] == FLOAT:
            return np.frombuffer(raw, dtype="<f4")
        if acc["componentType"] == UNSIGNED_INT:
            return np.frombuffer(raw, dtype="<u4")
        if acc["componentType"] == 5123:  # UNSIGNED_SHORT
            return np.frombuffer(raw, dtype="<u2").astype(np.int32)
        raise ValueError(f"componentType no soportado: {acc['componentType']}.")

    n = pos_acc["count"]
    m = idx_acc["count"]
    vertices = accessor_array(pos_acc).reshape(n, 3).astype(np.float32)
    normals = accessor_array(nrm_acc).reshape(n, 3).astype(np.float32)
    faces = accessor_array(idx_acc).reshape(m // 3, 3).astype(np.int32)

    name = doc["nodes"][0].get("name", "")
    return MeshData(vertices=vertices, normals=normals, faces=faces, name=name)


def save_model(glb_path: str | Path, mesh: MeshData, meta: ModelMeta) -> Path:
    """Guarda la malla como GLB y los metadatos en el sidecar ``*.muni.json``."""
    glb_path = Path(glb_path)
    write_glb(mesh, glb_path)
    meta.save(ModelMeta.sidecar_path(glb_path))
    return glb_path


def load_model(glb_path: str | Path) -> tuple[MeshData, Optional[ModelMeta]]:
    """Carga una malla GLB y, si existe, su sidecar de metadatos."""
    glb_path = Path(glb_path)
    mesh = read_glb(glb_path)
    sidecar = ModelMeta.sidecar_path(glb_path)
    meta = ModelMeta.load(sidecar) if sidecar.exists() else None
    return mesh, meta
