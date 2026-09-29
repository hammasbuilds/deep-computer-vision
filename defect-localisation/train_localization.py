"""Anomaly localisation on real MVTec-AD defects.

This is step 4 of anomalydiffusion's own pipeline (train a localisation model), run on
**real** anomalies rather than generated ones. That makes it the baseline their
generated data is supposed to beat, so it is worth having regardless of when the 5.73 GB
latent-diffusion checkpoint finishes downloading.

The network, optimiser, schedule and loss are taken from the repo's
`train-localization.py` so the comparison stays honest. Two things are deliberately
different:

1. **Checkpoint selection is on val, not test.** The repo calls `test()` every epoch and
   saves whenever the summed test metrics improve, which is model selection on the test
   set - the reported number becomes the best of N draws on the data it is scored
   against. Here val selects and test is read exactly once.
2. **The data is the real MVTec defect set** from `index.jsonl`, not generated pairs, so
   there is no `train/good` half and no synthetic augmentation.

Usage: see RUN.md in this folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..']*2), 'tools'))
from autotune import LRAutoTune  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "anomalydiffusion"))
from unet_utils.loss import FocalLoss  # noqa: E402
from unet_utils.model_unet import DiscriminativeSubNetwork  # noqa: E402

from dataroot import require as _ds  # noqa: E402
DATA = _ds("mvtec")
SIZE = 256


class DefectSet(Dataset):
    """Image + binary defect mask, resized to 256x256 as the repo's loader does."""

    def __init__(self, split: str, augment: bool = False):
        with open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") as fh:
            self.rows = [json.loads(line) for line in fh]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(os.path.join(DATA, r["image"])).convert("RGB").resize(
            (SIZE, SIZE), Image.BILINEAR
        )
        msk = Image.open(os.path.join(DATA, r["mask"])).convert("L").resize(
            (SIZE, SIZE), Image.NEAREST  # never interpolate a label map
        )
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        m = torch.from_numpy((np.asarray(msk, dtype=np.uint8) > 127).astype(np.float32))[None]

        if self.augment:
            if np.random.rand() < 0.5:
                x, m = torch.flip(x, [2]), torch.flip(m, [2])
            if np.random.rand() < 0.5:
                x, m = torch.flip(x, [1]), torch.flip(m, [1])

        return {"image": x, "mask": m, "object": r["object"], "defect": r["defect"]}


@torch.no_grad()
def evaluate(model, loader, device, per_object: bool = False) -> dict:
    """Pixel AP and IoU over a split, optionally per object.

    Pixel AP is the primary number: defect masks cover a median 1.47% of the frame, so
    accuracy would sit near 98.5% for a model that predicts all-background and would say
    nothing at all.

    Performance note. The first version of this built a Python list holding one object
    name *per pixel* - 75 images x 256 x 256 = 4.9M strings appended every epoch - and
    ran `average_precision_score` over all 4.9M points each time. The GPU sat at 0%
    utilisation while Python did that, so the evaluation, not the training, set the
    epoch time. Object identity is now tracked per image and expanded only when the
    per-object breakdown is actually wanted, which is at the end of a run rather than
    every epoch.
    """
    model.eval()
    probs, trues, obj_per_img = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        p = torch.softmax(model(x), dim=1)[:, 1:2]  # channel 1 = anomaly
        probs.append(p.cpu().numpy().reshape(x.shape[0], -1))
        trues.append(batch["mask"].numpy().reshape(x.shape[0], -1))
        obj_per_img.extend(batch["object"])

    p = np.concatenate(probs)                      # (n_images, pixels)
    t = np.concatenate(trues).astype(np.uint8)
    from sklearn.metrics import average_precision_score

    pf, tf = p.ravel(), t.ravel()
    out = {"pixel_ap": float(average_precision_score(tf, pf)) if tf.any() else float("nan")}
    pred = pf > 0.5
    inter = np.logical_and(pred, tf).sum()
    union = np.logical_or(pred, tf).sum()
    out["iou_pooled"] = float(inter / union) if union else float("nan")
    out["positive_rate"] = float(tf.mean())

    # Pooled IoU counts every pixel once across the whole split, so images with huge
    # masks supply most of the pixels and set the number almost by themselves. On this
    # data that is severe: `metal_nut/flip` covers ~48% of a frame against
    # `pill/color` at 0.17%. Measured on the test split, pooled IoU read 0.7899 while
    # the mean per-image IoU was 0.3770 and 7 of 26 images had *zero* overlap - every
    # screw defect type and both toothbrush images among them.
    #
    # Per-image IoU gives each image one vote, which is what "does it find the defect"
    # actually asks. Report both; they answer different questions and the gap between
    # them is itself the finding.
    pred_i = p > 0.5
    inter_i = np.logical_and(pred_i, t).sum(axis=1)
    union_i = np.logical_or(pred_i, t).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou_i = np.where(union_i > 0, inter_i / np.maximum(union_i, 1), np.nan)
    out["iou_per_image"] = float(np.nanmean(iou_i))
    out["zero_overlap_images"] = int((iou_i == 0).sum())
    out["n_images"] = int(len(iou_i))

    per = {}
    if per_object:
        objs = np.array(obj_per_img)
        for o in np.unique(objs):
            rows = objs == o                       # select whole images, not pixels
            if t[rows].any():
                per[str(o)] = float(average_precision_score(t[rows].ravel(), p[rows].ravel()))
    out["per_object_ap"] = per
    return out


def _require_cuda():
    # Abort rather than silently train on CPU. Installing anything that depends on
    # torch (facexlib, diffusers, transformers) pulls a CPU build into the workspace
    # venv, and PYTHONPATH is searched before the interpreter's own site-packages, so
    # that CPU build shadows the CUDA one. Training then runs ~50x slower with no
    # error - just a "pin_memory ... no accelerator found" warning in the log. That
    # cost two runs before it was spotted, so it is a hard failure now.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "  REFUSING TO TRAIN ON CPU: torch " + torch.__version__ +
            " reports no CUDA. A CPU torch is probably shadowing the CUDA build on "
            "PYTHONPATH. Check: python -c \"import torch;print(torch.__file__)\"")


def main() -> None:
    _require_cuda()
    ap = argparse.ArgumentParser()
    # This model is trained FROM SCRATCH - `DiscriminativeSubNetwork` starts from
    # `weights_init`, not a checkpoint - so it needs far more updates than a fine-tune.
    # A fine-tune here gets ~20k steps; from scratch gets 30k+. The repo's own default
    # of 200 epochs assumed its loader's `length=500` synthetic sampling per epoch, so
    # its "200" is ~12,500 steps, not 200 passes over a real 800-image set.
    #
    #   steps = epochs * N_train / batch
    # At N_train ~800 and batch 16 that is 50 steps/epoch, so 600 epochs = 30,000 steps.
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)           # measured best throughput
    ap.add_argument("--lr", type=float, default=5e-5)       # sweep winner at 30 epochs
    ap.add_argument("--workers", type=int, default=2)       # repo uses 16 (Linux)
    ap.add_argument("--save-every-epochs", type=int, default=5)
    # Validation runs average_precision_score over 190 x 65,536 = 12.4M points, which on
    # a shared GPU took longer than the 62 training steps it was measuring. Measured:
    # ~18.7 min/epoch with per-epoch validation, which turns 600 epochs into 7.8 days.
    # Validating every 5th epoch keeps the same best-checkpoint behaviour at a fraction
    # of the cost - the curve is smooth enough that 5-epoch resolution loses nothing.
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "checkpoints"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Never train on a dataset that is still being written. If a download is mid-flight
    # the split files are stale, images referenced by the index may not exist yet, and
    # the run silently trains on a subset while reporting the full count.
    n0 = len(os.listdir(DATA)) + sum(
        len(fs) for _, _, fs in os.walk(DATA))
    time.sleep(20)
    n1 = len(os.listdir(DATA)) + sum(
        len(fs) for _, _, fs in os.walk(DATA))
    if n1 != n0:
        raise SystemExit(f"  REFUSING TO TRAIN: {DATA} is still growing "
                         f"({n0} -> {n1} files in 20s). Wait for the download to finish.")
    print(f"  dataset stable at {n1} files", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407)
    np.random.seed(3407)

    # persistent_workers is not a micro-optimisation on Windows. Workers are spawned as
    # fresh processes that each re-import torch, and without this they are re-created
    # *every epoch*, for both loaders. Measured: 3 epochs took 6.1 min while the GPU
    # needed about 12s of work per epoch. The spawn cost, not the training, set the
    # epoch time.
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(DefectSet("train", augment=True), batch_size=args.bs, shuffle=True,
                    drop_last=True, pin_memory=True, **kw)
    va = DataLoader(DefectSet("val"), batch_size=args.bs, **kw)
    te = DataLoader(DefectSet("test"), batch_size=args.bs, **kw)

    model = DiscriminativeSubNetwork(in_channels=3, out_channels=2).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  DiscriminativeSubNetwork: {n_p/1e6:.2f} M params on {device}", flush=True)
    print(f"  train {len(tr.dataset)}  val {len(va.dataset)}  test {len(te.dataset)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(args.epochs * 0.8), int(args.epochs * 0.9)], gamma=0.2
    )
    focal = FocalLoss()

    best_ap, best_epoch, history = -1.0, -1, []
    start_epoch = 0
    last_path = os.path.join(args.out, "localization_last.pth")

    # Resume restores optimiser and scheduler too. Reloading weights alone would restart
    # the MultiStepLR at its initial rate and discard Adam's moment estimates, which
    # shows up as the model getting worse immediately after a resume.
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_ap, best_epoch = ck["best_ap"], ck["best_epoch"]
        history = ck.get("history", [])
        print(f"  RESUMED from epoch {start_epoch} (best val AP {best_ap:.4f} "
              f"at epoch {best_epoch})", flush=True)

        # A restored scheduler carries its own base_lr and rewrites the optimiser's lr
        # on every .step(), so a --lr given on a resume is silently discarded. That is
        # what kept MSRGAN on an old 1e-4 schedule (decayed to 5e-5) after asking for
        # 2e-4, and left it flat for 6,000 steps looking converged. Honour the caller.
        _restored = sched.base_lrs[0] if getattr(sched, "base_lrs", None) else None
        if _restored and abs(_restored - args.lr) / max(args.lr, 1e-12) > 0.01:
            for _g in opt.param_groups:
                _g["lr"] = args.lr
            sched =         torch.optim.lr_scheduler.MultiStepLR(
            opt, [int(args.epochs*0.8), int(args.epochs*0.9)], gamma=0.2)
            print(f"  LR OVERRIDE: restored base {_restored:.2e} -> using {args.lr:.2e}",
                  flush=True)

    elif args.resume:
        print("  --resume given but no localization_last.pth; starting fresh", flush=True)

    # checked every val cycle (every 5 epochs); raises the rate when
    # progress is real but slow, lowers it when the metric falls
    tuner = LRAutoTune(opt, good=0.002)
    t_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in tr:
            x = batch["image"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            out = torch.softmax(model(x), dim=1)
            loss = focal(out, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item()
            nb += 1
        sched.step()

        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            val = evaluate(model, va, device)
            history.append({"epoch": epoch, "train_loss": tot / max(nb, 1), **{
                k: v for k, v in val.items() if k != "per_object_ap"}})
            if val["pixel_ap"] > best_ap:
                best_ap, best_epoch = val["pixel_ap"], epoch
                torch.save(model.state_dict(), os.path.join(args.out, "localization_best.pth"))
        if do_val and (epoch % 5 == 0 or epoch == args.epochs - 1):
            print(f"  epoch {epoch:3d}  loss {tot/max(nb,1):.4f}  "
                  f"val AP {val['pixel_ap']:.4f}  IoU/img {val['iou_per_image']:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  best@{best_epoch}", flush=True)
            print(f"      autotune: {tuner.step(val['pixel_ap'])}", flush=True)

        # Rolling checkpoint every N epochs: one file, replaced each time, so disk stays
        # flat over a 600-epoch run. Written to a temp file and renamed, because
        # overwriting the only checkpoint in place means a crash mid-write loses
        # everything. `localization_best.pth` is separate and never rotated.
        if (epoch + 1) % args.save_every_epochs == 0 or epoch == args.epochs - 1:
            tmp = last_path + ".tmp"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "epoch": epoch,
                        "best_ap": best_ap, "best_epoch": best_epoch,
                        "history": history, "args": vars(args)}, tmp)
            os.replace(tmp, last_path)

    mins = (time.perf_counter() - t_start) / 60
    print(f"  trained {args.epochs} epochs in {mins:.1f} min; best val AP {best_ap:.4f} "
          f"at epoch {best_epoch}", flush=True)

    # Test is read once, with the val-selected checkpoint.
    model.load_state_dict(torch.load(os.path.join(args.out, "localization_best.pth")))
    res = {"train": evaluate(model, DataLoader(DefectSet("train"), batch_size=args.bs,
                                               num_workers=0),
                             device, per_object=True),
           "val": evaluate(model, va, device, per_object=True),
           "test": evaluate(model, te, device, per_object=True)}
    # Both IoUs, because they disagree sharply here and the gap is the finding: the
    # pooled figure is set by a few very large masks, the per-image one is not.
    print("\n  split    pixel-AP  IoU-pooled  IoU/image  zero-overlap  positive-rate")
    for k in ("train", "val", "test"):
        r = res[k]
        print(f"  {k:7s}  {r['pixel_ap']:.4f}     {r['iou_pooled']:.4f}     "
              f"{r['iou_per_image']:.4f}   {r['zero_overlap_images']:3d}/{r['n_images']:<4d}    "
              f"{r['positive_rate']*100:.2f}%")
    print("\n  per-object pixel-AP (test):")
    for o, v in sorted(res["test"]["per_object_ap"].items()):
        print(f"    {o:12s} {v:.4f}")

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "best_epoch": best_epoch, "best_val_ap": best_ap,
                   "history": history, "final": res, "minutes": mins}, fh, indent=2)
    print(f"\n  wrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
