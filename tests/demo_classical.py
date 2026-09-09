"""Demo paso a paso de ClassicalSegmenter con imagen real Golgi-Cox.

Carga el stack tests/fixtures/image-package-2/ y guarda cada paso de la
segmentación clásica como imagen PNG en tests/output_classical/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muni.core.volume import Volume3D
from muni.io.stack_loader import load_folder
from muni.preprocess.denoise import anisotropic_diffusion, normalize
from muni.segment.threshold import otsu_threshold, sigmoid_probability
from muni.segment.morphology import binary_open, binary_close
from muni.segment.connected import largest_component

FIXTURES = Path(__file__).parent / "fixtures" / "image-package-2"
OUT = Path(__file__).parent / "output_classical"
OUT.mkdir(exist_ok=True)


def save_slice(data: np.ndarray, title: str, filename: str, cmap: str = "gray", vmin=None, vmax=None):
    mid = data.shape[0] // 2
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(data[mid], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=120)
    plt.close(fig)
    print(f"  [{filename}] {title}")


def save_2x2(images, titles, filename, cmap="gray"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, img, title in zip(axes.flat, images, titles):
        mid = img.shape[0] // 2
        ax.imshow(img[mid], cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=120)
    plt.close(fig)
    print(f"  [{filename}] {titles}")


def main():
    print("=== ClassicalSegmenter demo ===")
    print(f"Cargando stack real desde: {FIXTURES}")
    vol, _ = load_folder(FIXTURES)
    print(f"Volumen: {vol.shape}\n")

    # --- Paso 1: Original ---
    save_slice(vol.data, "1. Original (Golgi-Cox real)", "01_original.png")

    # --- Paso 2: Normalizado ---
    normed = normalize(vol.data, (0.0, 255.0))
    save_slice(normed, "2. Normalizado [0-255]", "02_normalized.png")

    # --- Paso 3: Difusión anisotrópica ---
    denoised = anisotropic_diffusion(normed, iterations=5, kappa=15.0, gamma=0.2)
    save_slice(denoised, "3. Difusión anisotrópica (denoise)", "03_denoised.png")

    # --- Paso 4: Histograma + Otsu ---
    t = otsu_threshold(denoised)
    mid = denoised.shape[0] // 2
    flat = denoised[mid].ravel().astype(np.uint8)
    hist = np.bincount(flat, minlength=256)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(256), hist, color="steelblue", width=1.0)
    ax.axvline(t, color="red", linewidth=2, linestyle="--", label=f"Otsu t={t}")
    ax.set_title("4. Histograma + umbral Otsu")
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "04_histogram_otsu.png", dpi=120)
    plt.close(fig)
    print(f"  [04_histogram_otsu.png] Histograma + Otsu t={t}")

    # --- Paso 5: Mapa de probabilidad (sigmoide) ---
    prob = sigmoid_probability(denoised, t)
    save_slice(prob, "5. Probabilidad (sigmoide)", "05_probability.png", vmin=0, vmax=1)

    # --- Paso 6: Máscara binaria ---
    mask = prob >= 0.5
    save_slice(mask.astype(float), "6. Máscara binaria (prob >= 0.5)", "06_binary_mask.png", vmin=0, vmax=1)

    # --- Paso 7: Apertura morfológica ---
    opened = binary_open(mask, structure="cross", iterations=1)
    save_slice(opened.astype(float), "7. Apertura morfológica", "07_morph_open.png", vmin=0, vmax=1)

    # --- Paso 8: Cierre morfológico ---
    closed = binary_close(opened, structure="cross", iterations=1)
    save_slice(closed.astype(float), "8. Cierre morfológico", "08_morph_close.png", vmin=0, vmax=1)

    # --- Paso 9: Componente mayor ---
    final = largest_component(closed)
    save_slice(final.astype(float), "9. Componente mayor", "09_largest_component.png", vmin=0, vmax=1)

    # --- Resumen 2x2 ---
    save_2x2(
        [normed, denoised, prob, final.astype(float)],
        ["Normalizado", "Denoised", "Probabilidad", "Resultado final"],
        "10_summary.png",
    )

    print(f"\nListo. Imágenes guardadas en: {OUT}/")


if __name__ == "__main__":
    main()
