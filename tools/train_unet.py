"""Entrena el U-Net de segmentación usando pseudo-etiquetas del segmentador clásico.

El flujo (bootstrapping):
1. Carga los paquetes de imágenes (fixtures o carpetas indicadas).
2. Genera máscaras "pseudo-verdad" con :class:`ClassicalSegmenter`.
3. Entrena el U-Net 2.5D (3 cortes → probabilidad del central) con BCE.
4. Guarda los pesos para que :class:`UnetSegmenter` los cargue.

Uso:
    python tools/train_unet.py --epochs 8 --out models/unet_weights.pt

Nota: a partir de estas pseudo-etiquetas se pueden corregir manualmente algunos
cortes (anotación) y re-entrenar para mejorar la calidad del motor de IA.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def _load_pseudo_labels(folder: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Devuelve una lista de (volumen_norm, mascara) por paquete."""
    from muni.io.stack_loader import load_folder
    from muni.segment.classical import ClassicalSegmenter

    volume, _paths = load_folder(folder)
    seg = ClassicalSegmenter(denoise=True, keep_largest=True).segment(volume)

    f = np.asarray(volume.data, dtype=np.float32)
    lo, hi = f.min(), f.max()
    norm = (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)
    return [(norm, seg.mask)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folders", nargs="*", default=None, help="Carpetas con pilas")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--base", type=int, default=16, help="Canales base del U-Net")
    ap.add_argument("--out", default="models/unet_weights.pt")
    args = ap.parse_args(argv)

    try:
        import torch
        from torch import nn
    except ImportError:
        print("PyTorch no está instalado (instala el extra `[ai]`).", file=sys.stderr)
        return 1

    from muni.segment.unet import _build_unet_module

    fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    folders = args.folders or [
        str(fixtures / "image-package-1"),
        str(fixtures / "image-package-2"),
    ]

    data: list[tuple[np.ndarray, np.ndarray]] = []
    for folder in folders:
        print(f"Pseudo-etiquetas de {folder} ...", flush=True)
        data.extend(_load_pseudo_labels(Path(folder)))
    if not data:
        print("No hay datos.", file=sys.stderr)
        return 1

    # Dataset: para cada plano z, entrada = canales (z-h..z+h), etiqueta = plano z.
    half = args.channels // 2
    xs, ys = [], []
    for norm, mask in data:
        nz = norm.shape[0]
        for z in range(nz):
            stack = np.zeros((args.channels, norm.shape[1], norm.shape[2]), dtype=np.float32)
            for k in range(args.channels):
                zk = min(max(z + k - half, 0), nz - 1)
                stack[k] = norm[zk]
            xs.append(stack)
            ys.append(mask[z].astype(np.float32))
    X = torch.as_tensor(np.stack(xs))
    Y = torch.as_tensor(np.stack(ys)).unsqueeze(1)
    n = X.shape[0]
    print(f"Muestras: {n} ({[v.shape for v in [X, Y]]})", flush=True)

    # División train/val (80/20).
    idx = np.random.default_rng(0).permutation(n)
    n_train = int(0.8 * n)
    Xtr, Ytr = X[idx[:n_train]], Y[idx[:n_train]]
    Xva, Yva = X[idx[n_train:]], Y[idx[n_train:]]

    model = _build_unet_module(args.channels, args.base)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for b in range(0, n_train, args.batch):
            bi = perm[b : b + args.batch]
            preds = model(Xtr[bi])
            loss = criterion(preds, Ytr[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * bi.numel()
        train_loss = total / n_train

        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(Xva))
            inter = ((vp > 0.5) & (Yva > 0.5)).sum().item()
            union = ((vp > 0.5) | (Yva > 0.5)).sum().item()
            iou = inter / union if union else 1.0
        print(f"epoch {epoch+1}/{args.epochs} · loss={train_loss:.4f} · val_iou={iou:.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Pesos guardados en {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())