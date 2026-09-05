"""Segmentación con U-Net (PyTorch) — motor de calidad opcional.

Arquitectura U-Net implementada desde cero (sin librerías de segmentación ya
hechas). El ``UnetSegmenter`` usa una pila 2.5D: tres cortes consecutivos como
entrada para predecir la probabilidad del corte central.

PyTorch se importa de forma diferida: si no está instalado o faltan los pesos,
el segmentador levanta un error claro y la aplicación puede seguir usando el
motor clásico sin penalización.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from muni.core.volume import Volume3D
from muni.segment.base import Segmenter, SegmentationResult

# ---------------------------------------------------------------------------
# Definición de la red (PyTorch puro, sin torchtorchvision)
# ---------------------------------------------------------------------------


def _build_unet_module(in_channels: int, base: int) -> "nn.Module":
    """Construye un U-Net 2D con ``in_channels`` canales de entrada y 1 salida."""
    import torch
    from torch import nn

    class DoubleConv(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class Down(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.block = DoubleConv(cin, cout)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            skip = self.block(x)
            return self.pool(skip), skip

    class Up(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
            self.block = DoubleConv(cin, cout)

        def forward(self, x, skip):
            x = self.up(x)
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            return self.block(torch.cat([skip, x], dim=1))

    class Unet2D(nn.Module):
        def __init__(self, cin: int, base: int) -> None:
            super().__init__()
            b = base
            self.d1 = Down(cin, b)
            self.d2 = Down(b, b * 2)
            self.d3 = Down(b * 2, b * 4)
            self.bot = DoubleConv(b * 4, b * 8)
            self.u1 = Up(b * 8, b * 4)
            self.u2 = Up(b * 4, b * 2)
            self.u3 = Up(b * 2, b)
            self.head = nn.Conv2d(b, 1, 1)

        def forward(self, x):
            x, s1 = self.d1(x)
            x, s2 = self.d2(x)
            x, s3 = self.d3(x)
            x = self.bot(x)
            x = self.u1(x, s3)
            x = self.u2(x, s2)
            x = self.u3(x, s1)
            return self.head(x)

    return Unet2D(in_channels, base)


# ---------------------------------------------------------------------------
# Segmentador
# ---------------------------------------------------------------------------


class UnetSegmenter(Segmenter):
    """Segmenta con un U-Net 2.5D. Requiere ``torch`` y un archivo de pesos."""

    name = "unet"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        in_channels: int = 3,
        base_features: int = 16,
        threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.in_channels = in_channels
        self.base_features = base_features
        self.threshold = float(threshold)
        self._device = device
        self._model = None

    # ------------------------------------------------------------------ carga
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "El motor U-Net requiere PyTorch: instala el extra con "
                "`pip install muni[ai]`."
            ) from exc

        if self.model_path is None or not self.model_path.exists():
            raise RuntimeError(
                f"No se encontraron los pesos del U-Net en {self.model_path}. "
                "Entrena el modelo con tools/train_unet.py o usa el motor clásico."
            )

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_unet_module(self.in_channels, self.base_features)
        state = torch.load(self.model_path, map_location=device)
        model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
        model.to(device).eval()
        self._torch = torch
        self._device = device
        self._model = model

    # -------------------------------------------------------------- inferencia
    def segment(self, volume: Volume3D) -> SegmentationResult:
        self._ensure_model()
        torch = self._torch

        f = np.asarray(volume.data, dtype=np.float32)
        nz, ny, nx = f.shape
        if nz < self.in_channels:
            raise ValueError(
                f"El U-Net 2.5D necesita al menos {self.in_channels} planos."
            )

        # Entrada normalizada a [0, 1] con rechazo de extremos.
        lo, hi = f.min(), f.max()
        norm = (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)

        stack = np.zeros((nz, self.in_channels, ny, nx), dtype=np.float32)
        half = self.in_channels // 2
        for z in range(nz):
            for k in range(self.in_channels):
                zk = min(max(z + k - half, 0), nz - 1)
                stack[z, k] = norm[zk]

        prob = np.zeros((nz, ny, nx), dtype=np.float32)
        model = self._model
        device = self._device
        with torch.no_grad():
            for z in range(nz):
                inp = torch.from_numpy(stack[z : z + 1]).to(device)
                out = model(inp)
                prob[z] = torch.sigmoid(out).cpu().numpy()[0, 0]

        mask = prob >= self.threshold
        return SegmentationResult(
            probability=prob,
            mask=mask,
            threshold=self.threshold,
            method=self.name,
        )

    def batched_segment(self, volume: Volume3D, batch_size: int = 8) -> SegmentationResult:
        """Inferencia por lotes (menor sobrecarga de CPU/GPU)."""
        self._ensure_model()
        torch = self._torch

        f = np.asarray(volume.data, dtype=np.float32)
        nz, ny, nx = f.shape
        lo, hi = f.min(), f.max()
        norm = (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)

        stack = np.zeros((nz, self.in_channels, ny, nx), dtype=np.float32)
        half = self.in_channels // 2
        for z in range(nz):
            for k in range(self.in_channels):
                zk = min(max(z + k - half, 0), nz - 1)
                stack[z, k] = norm[zk]

        prob = np.zeros((nz, ny, nx), dtype=np.float32)
        model = self._model
        device = self._device
        with torch.no_grad():
            for start in range(0, nz, batch_size):
                end = min(start + batch_size, nz)
                inp = torch.from_numpy(stack[start:end]).to(device)
                out = torch.sigmoid(model(inp)).cpu().numpy()[:, 0]
                prob[start:end] = out

        mask = prob >= self.threshold
        return SegmentationResult(
            probability=prob,
            mask=mask,
            threshold=self.threshold,
            method=self.name,
        )