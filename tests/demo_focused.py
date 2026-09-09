"""Demo paso a paso de FocusedSegmenter con imagen real Golgi-Cox.

Carga el stack tests/fixtures/image-package-2/ y guarda cada paso de la
segmentación con filtro de nitidez como imagen PNG en tests/output_focused/.
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
from muni.segment.threshold import otsu_threshold, sigmoid_probability, _box_mean
from muni.segment.morphology import binary_open, binary_close
from muni.segment.connected import largest_component

FIXTURES = Path(__file__).parent / "fixtures" / "image-package-2"
OUT = Path(__file__).parent / "output_focused"
OUT.mkdir(exist_ok=True)


def local_std(image: np.ndarray, radius: int = 3) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    mean = _box_mean(img, radius)
    mean_sq = _box_mean(img * img, radius)
    variance = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(variance)


def save_slice(data, title, filename, cmap="gray", vmin=None, vmax=None):
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
    print("=== FocusedSegmenter demo ===")
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

    # --- Paso 5: Mapa de probabilidad ---
    prob = sigmoid_probability(denoised, t)
    save_slice(prob, "5. Probabilidad (sigmoide)", "05_probability.png", vmin=0, vmax=1)

    # --- Paso 6: Máscara base (Otsu) ---
    base_mask = prob >= 0.5
    save_slice(base_mask.astype(float), "6. Máscara base (Otsu)", "06_base_mask.png", vmin=0, vmax=1)

    # --- Paso 7: Mapa de nitidez (desviación estándar local) ---
    sharpness = np.zeros_like(denoised, dtype=np.float64)
    for z in range(denoised.shape[0]):
        sharpness[z] = local_std(denoised[z], radius=3)
    save_slice(sharpness, "7. Nitidez (std local por plano)", "07_sharpness.png", cmap="hot")

    # --- Paso 8: Umbral de nitidez ---
    s_thresh = float(np.percentile(sharpness[base_mask], 85)) if base_mask.any() else 0.0
    sharp_mask = sharpness > s_thresh
    save_slice(sharp_mask.astype(float), f"8. Máscara de nitidez (p50={s_thresh:.1f})", "08_sharp_mask.png", vmin=0, vmax=1)

    # --- Paso 9: Máscara combinada ---
    combined = base_mask & sharp_mask
    save_slice(combined.astype(float), "9. Combinada (base AND nitidez)", "09_combined.png", vmin=0, vmax=1)

    # --- Paso 10: Apertura ---
    opened = binary_open(combined, structure="cross", iterations=1)
    save_slice(opened.astype(float), "10. Apertura morfológica", "10_morph_open.png", vmin=0, vmax=1)

    # --- Paso 11: Cierre ---
    closed = binary_close(opened, structure="cross", iterations=1)
    save_slice(closed.astype(float), "11. Cierre morfológico", "11_morph_close.png", vmin=0, vmax=1)

    # --- Paso 12: Componente mayor ---
    final = largest_component(closed)
    save_slice(final.astype(float), "12. Componente mayor", "12_largest_component.png", vmin=0, vmax=1)

    # --- Resumen 2x2 ---
    save_2x2(
        [normed, denoised, prob, final.astype(float)],
        ["Normalizado", "Denoised", "Probabilidad", "Resultado final"],
        "13_summary.png",
    )

    print(f"\nListo. Imágenes guardadas en: {OUT}/")


if __name__ == "__main__":
    main()
