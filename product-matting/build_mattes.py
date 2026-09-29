"""Derive alpha mattes for the MVTec product images, to fine-tune u2netp.

MVTec objects sit on near-uniform backgrounds, so a matte can be recovered without any
hand annotation. These are **weak labels**, not ground truth, and the script is written
to make that visible rather than hide it:

  * the background colour is estimated from the image border, not assumed to be white
  * every matte gets a confidence score, and low-scoring ones are rejected outright
  * the rejection rate is printed, because a low one would mean the filter is not
    working rather than that the data is perfect

Rejecting aggressively matters more than the final count. A wrong matte teaches the
model to cut in the wrong place, and there is no later step that catches it.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "3-defect-generation", "data")
DST = os.path.join(HERE, "data")
SIZE = 320


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions not connected to the image border.

    Flood-fills the background inward from the border; whatever background the flood
    cannot reach is enclosed by the object and is therefore a hole. Implemented with a
    scanline queue so the module needs no scipy.
    """
    h, w = mask.shape
    bg = ~mask
    reach = np.zeros_like(bg)
    stack = []
    for x in range(w):
        if bg[0, x]:
            stack.append((0, x))
        if bg[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if bg[y, 0]:
            stack.append((y, 0))
        if bg[y, w - 1]:
            stack.append((y, w - 1))

    while stack:
        y, x = stack.pop()
        if reach[y, x] or not bg[y, x]:
            continue
        reach[y, x] = True
        # extend along the row, then push the rows above and below
        x0 = x
        while x0 > 0 and bg[y, x0 - 1] and not reach[y, x0 - 1]:
            x0 -= 1
            reach[y, x0] = True
        x1 = x
        while x1 < w - 1 and bg[y, x1 + 1] and not reach[y, x1 + 1]:
            x1 += 1
            reach[y, x1] = True
        for xi in range(x0, x1 + 1):
            if y > 0 and bg[y - 1, xi] and not reach[y - 1, xi]:
                stack.append((y - 1, xi))
            if y < h - 1 and bg[y + 1, xi] and not reach[y + 1, xi]:
                stack.append((y + 1, xi))

    return mask | (bg & ~reach)


def estimate_matte(img: Image.Image) -> tuple[np.ndarray, float]:
    """Return (alpha in [0,1], confidence).

    Confidence is the separation between object and background in the distance-to-
    background-colour histogram. A bimodal image (object clearly unlike its background)
    scores high; a low score means the object blends in and the matte is unreliable.
    """
    im = img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)

    # Background colour from a border ring - never assume white.
    ring = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3),
                           a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)

    dist = np.linalg.norm(a - bg, axis=2)
    dist /= max(dist.max(), 1e-6)

    # Otsu threshold on the distance map.
    hist, edges = np.histogram(dist, bins=64, range=(0, 1))
    w = hist.cumsum()
    mids = (edges[:-1] + edges[1:]) / 2
    mu = (hist * mids).cumsum()
    tot_w, tot_mu = w[-1], mu[-1]
    # Between-class variance. The degenerate bins - where one class is empty - must be
    # excluded, not merely regularised: at the final bin `w == tot_w`, so the
    # denominator is zero and an epsilon guard makes the ratio explode, which sends
    # argmax to that bin every time. The threshold then lands above every pixel and
    # every matte comes out empty. (It did: 504 of 504 rejected.)
    # Between-class variance, unnormalised form:
    #     (tot_mu * w - mu * tot_w)^2 / (w * (tot_w - w))
    # The `* tot_w` on the first moment is not optional. Dropping it (as a first version
    # here did) leaves a term that grows with k, so argmax lands on the last bin, the
    # threshold sits above almost every pixel, and every matte comes out empty - 504 of
    # 504 rejected, with the failure looking like "the data is bad".
    # Bins where one class is empty are excluded outright rather than regularised.
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = w * (tot_w - w)
        num = (tot_mu * w - mu * tot_w) ** 2
        between = np.where(denom > 0, num / np.maximum(denom, 1), -np.inf)
    k = int(np.nanargmax(between))
    thr = mids[k]

    alpha = (dist > thr).astype(np.float32)

    # Close interior holes. The measure is "distance from the border colour", and MVTec
    # backgrounds are dark, so a *dark region inside the object* looks exactly like
    # background: the raw matte punches holes through the metal_nut's textured centre
    # and the toothbrush's bristles. A product is a solid object, so any background
    # region not connected to the image border is a hole, not background.
    #
    # This is not cosmetic. Training a matting model on holed labels teaches it to
    # punch holes through products, and nothing downstream would catch that.
    raw = alpha > 0.5          # keep the pre-fill mask: confidence must measure colour
    alpha = _fill_holes(raw).astype(np.float32)   # separability, not post-processing

    # Confidence: how well separated the two modes are, 0..1.
    fg, bgm = dist[raw], dist[~raw]
    if fg.size < 100 or bgm.size < 100:
        return alpha, 0.0
    sep = (fg.mean() - bgm.mean()) / (fg.std() + bgm.std() + 1e-6)
    conf = float(np.clip(sep / 4.0, 0, 1))

    # Soften the edge so the matte is not a hard binary cut - u2netp predicts soft alpha.
    alpha_img = Image.fromarray((alpha * 255).astype("uint8")).filter(
        ImageFilter.GaussianBlur(1.2))
    return np.asarray(alpha_img, dtype=np.float32) / 255.0, conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--min-fg", type=float, default=0.02)
    ap.add_argument("--max-fg", type=float, default=0.80)
    args = ap.parse_args()

    with open(os.path.join(SRC, "index.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]

    os.makedirs(os.path.join(DST, "images"), exist_ok=True)
    os.makedirs(os.path.join(DST, "mattes"), exist_ok=True)

    kept, rejected = [], {"conf": 0, "fg_small": 0, "fg_large": 0}
    for r in rows:
        img = Image.open(os.path.join(SRC, r["image"]))
        alpha, conf = estimate_matte(img)
        fg = float((alpha > 0.5).mean())
        if conf < args.min_conf:
            rejected["conf"] += 1;  continue
        if fg < args.min_fg:
            rejected["fg_small"] += 1;  continue
        if fg > args.max_fg:
            rejected["fg_large"] += 1;  continue

        name = f"{len(kept):05d}"
        img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(
            os.path.join(DST, "images", f"{name}.png"))
        Image.fromarray((alpha * 255).astype("uint8")).save(
            os.path.join(DST, "mattes", f"{name}.png"))
        kept.append({"name": name, "object": r["object"], "conf": round(conf, 4),
                     "fg": round(fg, 4)})

    n_rej = sum(rejected.values())
    print(f"  input {len(rows)}  kept {len(kept)}  rejected {n_rej} "
          f"({100*n_rej/max(len(rows),1):.1f}%)")
    print(f"    low confidence : {rejected['conf']}")
    print(f"    foreground <{args.min_fg:.0%} : {rejected['fg_small']}")
    print(f"    foreground >{args.max_fg:.0%} : {rejected['fg_large']}")
    if n_rej == 0:
        print("    WARNING: nothing rejected - the filter is probably not working")

    by = {}
    for k in kept:
        by.setdefault(k["object"], []).append(k)
    print("  per object:")
    for o, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"    {o:12s} {len(v):4d}  mean conf {np.mean([x['conf'] for x in v]):.3f}")

    # 80/15/5, stratified by object, seed 3407 - the house split policy.
    rng = random.Random(3407)
    splits = {"train": [], "val": [], "test": []}
    for o, group in sorted(by.items()):
        g = group[:]; rng.shuffle(g)
        n = len(g); n_test = max(1, round(n * 0.05)); n_val = max(1, round(n * 0.15))
        splits["test"] += g[:n_test]
        splits["val"] += g[n_test:n_test + n_val]
        splits["train"] += g[n_test + n_val:]
    for name, rs in splits.items():
        with open(os.path.join(DST, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
    tot = sum(len(v) for v in splits.values())
    print("  splits: " + "  ".join(
        f"{k} {len(v)} ({100*len(v)/max(tot,1):.1f}%)" for k, v in splits.items()))


if __name__ == "__main__":
    main()
