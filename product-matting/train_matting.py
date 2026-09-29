"""Fine-tune a background-removal (matting) model on the 1,337-pair set.

Why not u2netp directly: rembg ships u2netp as an **ONNX export**, which runs but cannot
be trained. Training needs the PyTorch weights from the U-2-Net repo
(https://github.com/xuebinqin/U-2-Net), which are on Google Drive.

So the script supports two backbones and picks whichever is available:

  u2netp   the real thing, if `_weights/u2netp.pth` is present (4.7 MB)
  unet     smp U-Net with a ResNet-34 encoder, ImageNet-pretrained if
           `_weights/resnet34-b627a593.pth` is present, otherwise from scratch

Either way the zero-shot u2netp ONNX result stays the baseline to beat, so the
comparison is honest regardless of which backbone trains.

Parameters are derived from the dataset, not copied:
    steps = epochs * N_train / batch
N_train is 1,070, so at batch 16 that is 67 steps/epoch and a 7,000-step budget is
about 105 epochs.

Labels are mixed: 1,000 LIP-ATR masks are human-annotated, 337 MVTec mattes are derived
by background subtraction. Metrics are reported per `label_source` as well as pooled,
because a model can score well on derived labels merely by reproducing the derivation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
from dataroot import require as _ds  # noqa: E402
DATA = _ds("product-mattes")
WEIGHTS = os.path.join(HERE, "..", "..", "_weights")
SIZE = 320


class MatteSet(Dataset):
    def __init__(self, split: str, augment: bool = False):
        with open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") as fh:
            self.rows = [json.loads(line) for line in fh]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(os.path.join(DATA, "images", r["name"] + ".png")).convert("RGB")
        m = Image.open(os.path.join(DATA, "mattes", r["name"] + ".png")).convert("L")
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.BILINEAR)
            m = m.resize((SIZE, SIZE), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy((np.asarray(m, np.float32) / 255.0))[None]
        if self.augment and random.random() < 0.5:
            x, y = torch.flip(x, [2]), torch.flip(y, [2])
        return x, y, r.get("label_source", "unknown")


def build(device: str):
    """Prefer the real u2netp; fall back to smp U-Net. Reports which, loudly."""
    # Either checkpoint from the U-2-Net repo works. They are DIFFERENT architectures,
    # not two sizes of one: u2netp.pth is U2NETP (4.7 MB, the small variant matching the
    # u2netp.onnx already used for inference) and u2net.pth is U2NET (176 MB, full).
    # Loading one into the other's class fails on every key, so the class is chosen by
    # filename rather than guessed.
    import sys
    sys.path.insert(0, os.path.join(HERE, "U-2-Net"))
    for fname, cls_name, label in (("u2netp.pth", "U2NETP", "u2netp"),
                                   ("u2net.pth", "U2NET", "u2net")):
        path = os.path.join(WEIGHTS, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 1e6:
            continue
        try:
            import model as u2mod
            m = getattr(u2mod, cls_name)(3, 1)
            sd = torch.load(path, map_location="cpu", weights_only=False)
            m.load_state_dict(sd, strict=True)   # strict: a silent mismatch trains noise
            print(f"  backbone: {label} (real PyTorch weights, "
                  f"{os.path.getsize(path)/2**20:.1f} MB, "
                  f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M params)")
            return m.to(device), label
        except Exception as e:
            print(f"  {fname} present but unusable ({str(e)[:90]}); trying next")

    import segmentation_models_pytorch as smp
    enc = os.path.join(WEIGHTS, "resnet34-b627a593.pth")
    weights = None
    if os.path.exists(enc) and os.path.getsize(enc) > 8e7:
        os.environ.setdefault("TORCH_HOME", os.path.abspath(os.path.join(WEIGHTS, "..")))
        weights = "imagenet"
    m = smp.Unet(encoder_name="resnet34", encoder_weights=weights,
                 in_channels=3, classes=1)
    print(f"  backbone: smp U-Net resnet34, encoder_weights={weights or 'None (scratch)'}")
    return m.to(device), "unet_resnet34"


def forward(model, kind: str, x):
    out = model(x)
    # U2NETP returns 7 side outputs; the first is the fused prediction.
    return out[0] if isinstance(out, (tuple, list)) else out


def probs(model, kind: str, x):
    """Foreground probability in [0,1], whichever backbone is in use.

    The two backbones disagree about what forward() returns and the code treated them
    as if they agreed:

        U2NET / U2NETP   applies sigmoid INSIDE its own forward - already probabilities
        smp.Unet         returns raw logits

    Everything downstream then applied sigmoid unconditionally. On the u2net path that
    is a second sigmoid, which maps [0,1] onto [sigmoid(0), sigmoid(1)] = [0.500, 0.731]
    - measured on the trained checkpoint, and exactly the grey mattes it was producing.
    The model could not express a confident prediction at all, and BCE-with-logits
    applied a third sigmoid during training, so the gradients were squashed too.

    IoU hid it completely, because thresholding at 0.5 sits right at the squashed
    background level - which is why IoU looked healthy at 0.82 while MAE sat frozen at
    0.4464 across three checkpoints.
    """
    out = forward(model, kind, x)
    if kind.startswith("u2net"):
        return out.clamp(0.0, 1.0)
    return torch.sigmoid(out)


@torch.no_grad()
def evaluate(model, kind, loader, device) -> dict:
    model.eval()
    inter = union = 0.0
    per_src: dict = {}
    maes = []
    for x, y, srcs in loader:
        x, y = x.to(device), y.to(device)
        p = probs(model, kind, x)
        maes.append(float(torch.mean(torch.abs(p - y))))
        pb, yb = (p > 0.5).float(), (y > 0.5).float()
        i = float((pb * yb).sum()); u = float(((pb + yb) > 0).float().sum())
        inter += i; union += u
        for j, s in enumerate(srcs):
            ii = float((pb[j] * yb[j]).sum()); uu = float(((pb[j] + yb[j]) > 0).float().sum())
            d = per_src.setdefault(s, [0.0, 0.0])
            d[0] += ii; d[1] += uu
    return {"iou": inter / max(union, 1e-9), "mae": float(np.mean(maes)),
            "per_source_iou": {k: v[0] / max(v[1], 1e-9) for k, v in per_src.items()}}


@torch.no_grad()
def save_inference(model, kind, device, out_dir: str, tag: str, k: int = 4) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(DATA, "test.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    seen, pick = set(), []
    for r in rows:                       # one per label source, then fill
        if r["label_source"] not in seen:
            seen.add(r["label_source"]); pick.append(r)
    pick += [r for r in rows if r not in pick][:max(0, k - len(pick))]
    model.eval()
    for r in pick[:k]:
        n = r["name"]
        img = Image.open(os.path.join(DATA, "images", n + ".png")).convert("RGB")
        x = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.)[None].to(device)
        p = probs(model, kind, x)[0, 0].cpu().numpy()
        a = Image.fromarray((p * 255).astype("uint8"))
        rgba = img.convert("RGBA"); rgba.putalpha(a)
        img.save(os.path.join(out_dir, f"{n}_input.png"))
        a.save(os.path.join(out_dir, f"{n}_{tag}_alpha.png"))
        rgba.save(os.path.join(out_dir, f"{n}_{tag}_cutout.png"))
        Image.open(os.path.join(DATA, "mattes", n + ".png")).save(
            os.path.join(out_dir, f"{n}_ground_truth.png"))


def save_rolling(path: str, payload: dict) -> None:
    """One rolling checkpoint, written via temp + atomic rename so an interrupted save
    cannot destroy the only copy."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


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
    ap.add_argument("--epochs", type=int, default=105)     # ~7000 steps at N=1070, bs 16
    ap.add_argument("--total", type=int, default=0,
                    help="whole budget, so the cosine spans the run not the chunk")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every-epochs", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--decay-lr", action="store_true",
                    help="opt back in to cosine decay; OFF by default - the rate is held")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    # Never train on a dataset that is still being written - a mid-flight download means
    # stale splits and a run that silently uses a subset while reporting the full count.
    n0 = sum(len(fs) for _, _, fs in os.walk(DATA))
    time.sleep(20)
    n1 = sum(len(fs) for _, _, fs in os.walk(DATA))
    if n1 != n0:
        raise SystemExit(f"  REFUSING TO TRAIN: {DATA} is still growing "
                         f"({n0} -> {n1} files in 20s). Wait for the download to finish.")
    print(f"  dataset stable at {n1} files", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407); np.random.seed(3407); random.seed(3407)
    os.makedirs(args.out, exist_ok=True)

    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    tr = DataLoader(MatteSet("train", True), batch_size=args.bs, shuffle=True,
                    drop_last=True, pin_memory=True, **kw)
    va = DataLoader(MatteSet("val"), batch_size=args.bs, **kw)
    te = DataLoader(MatteSet("test"), batch_size=args.bs, **kw)

    model, kind = build(device)

    # U2NET is 44 M params with 7 side outputs; a measured batch-8 training step peaks at
    # 7.7 GiB, so batch 16 would ask ~15 GiB of a 16 GiB card. On this card that does NOT
    # raise OutOfMemoryError - WDDM spills into host memory and the step runs ~10x slower
    # with no error at all - so an over-large batch would quietly waste the whole night.
    # Cap it here rather than trusting the flag.
    if kind in ("u2net",) and args.bs > 8:
        print(f"  batch {args.bs} -> 8 (U2NET at 320px measured 7.7 GiB at bs 8; "
              f"larger silently spills to host memory on this GPU)", flush=True)
        args.bs = 8
        kw2 = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
        tr = DataLoader(MatteSet("train", True), batch_size=args.bs, shuffle=True,
                        drop_last=True, pin_memory=True, **kw2)
        va = DataLoader(MatteSet("val"), batch_size=args.bs, **kw2)
        te = DataLoader(MatteSet("test"), batch_size=args.bs, **kw2)

    spe = max(1, len(tr.dataset) // args.bs)
    print(f"  N_train {len(tr.dataset)}  batch {args.bs}  {spe} steps/epoch "
          f"-> {args.epochs} epochs = {args.epochs*spe} steps  lr {args.lr}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # T_max is the WHOLE budget, not this chunk's target. The queue trains in 5-epoch
    # chunks and passes the chunk end as --epochs, so a cosine built on that completed
    # inside every chunk and landed on eta_min at its end. The queue then read that
    # floor out of the checkpoint and passed it as the next chunk's --lr, so the rate
    # stuck at the floor permanently: gfpgan ran its last 7,500 steps - half its budget
    # - at 1e-7, its autotuner reporting "healthy, lr held" the whole time because the
    # gain was positive and it had no idea the schedule had bottomed out.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.total or args.epochs), eta_min=1e-5)
    last_path = os.path.join(args.out, f"matting_{kind}_last.pth")
    best_path = os.path.join(args.out, f"matting_{kind}_best.pth")

    start_ep = 0
    best = {"iou": -1.0, "epoch": -1}
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        try:
            sched.load_state_dict(ck["scheduler"])
        except KeyError:
            # The scheduler state is from a different scheduler type. Runs that
            # started under CosineAnnealingLR save no "lr_lambdas", so restoring
            # them into the constant LambdaLR raises KeyError and killed
            # sr_msrgan at 5,000 of 20,000 steps. The schedule is rebuilt below
            # anyway, so the stored state is not needed - only the weights and
            # the optimiser are.
            print("  scheduler state is from a different scheduler type; "
                  "rebuilding instead of restoring", flush=True); start_ep = ck["epoch"] + 1; best = ck["best"]
        print(f"  RESUMED from epoch {start_ep} (best IoU {best['iou']:.4f})", flush=True)

        # A restored optimiser carries the DECAYED rate, not the base rate. The override
        # below compares sched.base_lrs, which still reads the original 2e-4 and
        # therefore matches --lr, so it never fires - while the optimiser is actually
        # running at whatever the cosine had decayed to. colourisation resumed and
        # trained at 1e-6, two hundred times below the floor, with the autotuner
        # reporting "healthy, lr held" because the metric happened to be rising.
        _cur = opt.param_groups[0]["lr"]
        if _cur < 1e-5:
            for _g in opt.param_groups:
                _g["lr"] = max(args.lr, 1e-5)
            print(f"  LR FLOOR: restored {_cur:.2e} is below the 1e-5 floor -> "
                  f"{opt.param_groups[0]['lr']:.2e}", flush=True)

        # A restored scheduler carries its own base_lr and rewrites the optimiser's lr
        # on every .step(), so a --lr given on a resume is silently discarded. That is
        # what kept MSRGAN on an old 1e-4 schedule (decayed to 5e-5) after asking for
        # 2e-4, and left it flat for 6,000 steps looking converged. Honour the caller.
        _restored = sched.base_lrs[0] if getattr(sched, "base_lrs", None) else None
        if _restored and abs(_restored - args.lr) / max(args.lr, 1e-12) > 0.01:
            for _g in opt.param_groups:
                _g["lr"] = args.lr
            sched =         torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs - start_ep), eta_min=1e-5)
            print(f"  LR OVERRIDE: restored base {_restored:.2e} -> using {args.lr:.2e}",
                  flush=True)


    before = evaluate(model, kind, va, device)
    print(f"  BEFORE: val IoU {before['iou']:.4f}  MAE {before['mae']:.4f}", flush=True)
    save_inference(model, kind, device, os.path.join(args.out, "inference_matting"), "before")

    # Checked every validation cycle: raises the rate when progress is
    # real but slow, lowers it when the metric falls.
    tuner = LRAutoTune(opt, good=0.002)

    if not args.decay_lr:
        # Measured on colourisation: the cosine restarts at the chunk's high rate every
        # time the queue resumes, which COST 0.25 dB per chunk before the decay brought
        # it back. Mean gain per epoch-block by rate: 3.63e-4 -0.2484, 2e-4 +0.0997,
        # 1e-5 +0.2832, 1e-6 +0.0225. Almost all of each chunk was spent recovering from
        # damage the high rate had just done. Holding the best measured rate removes the
        # restart entirely.
        for _g in opt.param_groups:
            _g["lr"] = args.lr
            # A previous CosineAnnealingLR leaves "initial_lr" behind in every param
            # group, and LambdaLR takes THAT as its base rather than the current lr. So
            # the constant schedule multiplied the old 2e-4 base by 1.0 and pinned the
            # run at 2e-4 while printing "LR HELD AT 1.00e-05" - true when printed, false
            # one line later. Clear it so the base is the rate actually asked for.
            _g.pop("initial_lr", None)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        tuner = None
        print(f"  LR HELD AT {args.lr:.2e} for the whole chunk "
              f"(no cosine, no per-epoch autotune; the queue changes it between chunks only when the metric says so)",
              flush=True)
    hist = []
    t0 = time.perf_counter()
    for ep in range(start_ep, args.epochs):
        model.train(); tot = nb = 0
        for x, y, _ in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # BCE + soft-dice: BCE alone under-segments when the foreground is a
            # minority of pixels, dice alone is unstable early. BCE on probabilities,
            # not with_logits, because probs() has already applied the activation -
            # with_logits would apply another one.
            p = probs(model, kind, x)
            dice = 1 - (2*(p*y).sum() + 1) / ((p+y).sum() + 1)
            loss = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y) + dice
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        v = evaluate(model, kind, va, device)
        hist.append({"epoch": ep, "loss": tot/max(nb,1), **{k: v[k] for k in ("iou","mae")}})
        flag = ""
        if v["iou"] > best["iou"]:
            best = {"iou": v["iou"], "epoch": ep}
            torch.save({"model": model.state_dict(), "kind": kind}, best_path); flag = "  *best"
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}/{args.epochs}  loss {tot/max(nb,1):.4f}  "
                  f"val IoU {v['iou']:.4f}  MAE {v['mae']:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}{flag}", flush=True)
            if tuner is not None:
                print(f"      autotune: {tuner.step(v['iou'])}", flush=True)
        if (ep + 1) % args.save_every_epochs == 0 or ep == args.epochs - 1:
            save_rolling(last_path, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                     "scheduler": sched.state_dict(), "epoch": ep,
                                     "best": best, "history": hist, "args": vars(args)})

    mins = (time.perf_counter() - t0) / 60
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    res = {"train": evaluate(model, kind, DataLoader(MatteSet("train"), batch_size=args.bs), device),
           "val": evaluate(model, kind, va, device),
           "test": evaluate(model, kind, te, device)}
    save_inference(model, kind, device, os.path.join(args.out, "inference_matting"), "after")

    print(f"\n  trained {args.epochs-start_ep} epochs in {mins:.1f} min; "
          f"best val IoU {best['iou']:.4f} at epoch {best['epoch']}")
    print(f"  {'split':7s} {'IoU':>8s} {'MAE':>8s}   per-source IoU")
    for s in ("train", "val", "test"):
        ps = "  ".join(f"{k}={v:.3f}" for k, v in sorted(res[s]["per_source_iou"].items()))
        print(f"  {s:7s} {res[s]['iou']:8.4f} {res[s]['mae']:8.4f}   {ps}")
    gain = res["val"]["iou"] - before["iou"]
    print(f"  BEFORE val IoU {before['iou']:.4f} -> AFTER {res['val']['iou']:.4f} = {gain:+.4f}")
    if gain < 0.005:
        print(f"  NOT SATISFIED. Continue with:")
        print(f"    python train_matting.py --resume --epochs {args.epochs*2} --lr {args.lr/2:.1e}")

    with open(os.path.join(args.out, "results_matting.json"), "w", encoding="utf-8") as fh:
        json.dump({"backbone": kind, "n_train": len(tr.dataset), "batch": args.bs,
                   "epochs": args.epochs, "steps": args.epochs*spe, "lr": args.lr,
                   "minutes": round(mins, 2), "before_val": before, "final": res,
                   "history": hist}, fh, indent=2)


if __name__ == "__main__":
    main()
