"""Build the README comparison figures from the committed sample outputs.

Nothing is re-run here: every panel is a PNG already committed under a project's
``samples/`` folder, and the per-image scores are computed from those PNGs.

    python docs/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def iou(pred: np.ndarray, truth: np.ndarray, thr: int = 128) -> float:
    p, t = pred >= thr, truth >= thr
    union = np.logical_or(p, t).sum()
    return float(np.logical_and(p, t).sum() / union) if union else 1.0


def mae(alpha: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(alpha.astype(np.float64) - truth.astype(np.float64)).mean() / 255.0)


def on_checker(rgba: np.ndarray) -> np.ndarray:
    """Composite a cut-out over a grey checkerboard so transparency is visible."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // 16) + (xx // 16)) % 2 == 0, 225, 165).astype(np.float64)
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    return (rgba[..., :3] * a + board[..., None] * (1 - a)).astype(np.uint8)


def grid(rows, columns, title, out):
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(2.6 * len(columns), 2.7 * len(rows)))
    for r, (label, cells) in enumerate(rows):
        for c, (img, caption) in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            top = columns[c] if r == 0 else ""
            ax.set_title("\n".join(x for x in (top, caption) if x), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"Sr {r + 1}\n{label}", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def defect_figure():
    s = ROOT / "defect-localisation" / "samples"
    picks = [
        ("pill_color", "pill, colour spot"),
        ("tile_crack", "tile, crack"),
        ("metal_nut_bent", "metal nut, bent"),
        ("wood_color", "wood, colour stain"),
    ]
    rows = []
    for stem, label in picks:
        truth = load(s / f"{stem}_ground_truth.png")
        mask = load(s / f"{stem}_pred_mask.png")
        rows.append((label, [
            (load(s / f"{stem}_input.png"), ""),
            (truth, ""),
            (load(s / f"{stem}_pred_heatmap.png"), ""),
            (mask, f"IoU {iou(mask, truth):.3f}"),
            (load(s / f"{stem}_overlay.png"), ""),
        ]))
    grid(rows, ["Input", "Ground truth", "Predicted heatmap", "Predicted mask", "Overlay"],
         "Defect localisation on MVTec AD images (IoU of the predicted mask against the ground truth)",
         OUT / "defect_localisation.png")


def matting_figure():
    # The samples/ folder also holds *_before_* files. They are not shown: the training
    # script writes them when a run starts or resumes, and this run resumed (its
    # before_val IoU 0.9621 equals the epoch-55 history entry), so they are a mid-run
    # checkpoint, not the published U2-Net weights.
    s = ROOT / "product-matting" / "samples"
    picks = ["00210", "01069", "01132", "01327"]
    rows = []
    for stem in picks:
        truth = load(s / f"{stem}_ground_truth.png")
        after = load(s / f"{stem}_after_alpha.png")
        rows.append((f"image {stem}", [
            (load(s / f"{stem}_input.png"), ""),
            (truth, ""),
            (after, f"IoU {iou(after, truth):.3f} / MAE {mae(after, truth):.4f}"),
            (on_checker(load(s / f"{stem}_after_cutout.png")), ""),
        ]))
    grid(rows, ["Input", "Ground-truth matte", "Predicted matte", "Cut-out"],
         "Product matting, fine-tuned U2-Net on held-out test images (IoU / MAE against the ground truth)",
         OUT / "product_matting.png")


if __name__ == "__main__":
    defect_figure()
    matting_figure()
