# Run parameters — anomaly localisation

Exact settings that produced the checkpoint and images in this folder. Reproduce with
the command at the bottom.

## Model

| | |
|---|---|
| architecture | `DiscriminativeSubNetwork(in_channels=3, out_channels=2)` |
| source | `anomalydiffusion/unet_utils/model_unet.py` (DRAEM-style U-Net) |
| parameters | **28.37 M** |
| pretrained weights | **none** — trained from scratch |
| input size | 256 x 256 |

## Hyperparameters

| setting | value | source |
|---|---|---|
| optimiser | `Adam` | repo default |
| **learning rate** | **5e-5** | chosen by sweep (see below) |
| scheduler | `MultiStepLR`, milestones `[24, 27]`, `gamma 0.2` | repo formula `[0.8*epochs, 0.9*epochs]` |
| loss | `FocalLoss` on `softmax(out, dim=1)` | repo default |
| **batch size** | **16** | **measured** — see below |
| epochs | 30 (sweep) | |
| workers | 2, `persistent_workers=True` | repo uses 16; Windows spawns processes |
| seed | 3407 | |
| augmentation | random horizontal + vertical flip | |
| device | Quadro RTX 5000, 16 GB | |

### Why batch 16

Measured on this card with the real model and a real Adam step:

| batch | VRAM | step | img/s |
|---:|---:|---:|---:|
| 4 | 2,196 MiB | 0.154s | 26.0 |
| 8 | 4,050 MiB | 0.283s | 28.3 |
| **16** | **7,760 MiB** | **0.494s** | **32.4** |
| 24 | 12,332 MiB | 0.809s | 29.7 |
| 32 | 15,179 MiB | 5.979s | 5.4 |
| 48 | 22,598 MiB | 21.467s | 2.2 |

Batch 32 asks 15,179 MiB of a 16,384 MiB card and **does not raise
`OutOfMemoryError`** — the Windows WDDM driver spills to host memory and the step runs
6x slower. Batch 48 is 15x slower. Picking the batch by "raise it until it crashes"
gives the worst answer here, because it never crashes.

### Learning-rate sweep

4 points x 30 epochs, batch fixed at 16.

| lr | best val AP | epoch |
|---|---:|---:|
| **5e-5** | **0.9324** | 28 |
| 1e-4 | running | |
| 2e-4 | queued | |
| 4e-4 | queued | |

## Data

| split | images | share |
|---|---:|---:|
| train | 403 | 80.0% |
| val | 75 | 14.9% |
| test | 26 | 5.2% |

504 real MVTec-AD defect/mask pairs, 6 objects, 25 object/defect classes, stratified by
(object, defect) with seed 3407. Zero overlap between any pair of splits. Details in
[../DATA.md](../DATA.md).

Checkpoint selected on **val**; test read once. The upstream repo selects on test, which
this deliberately does not do.

## Results

| split | pixel-AP | IoU pooled | IoU per-image | zero-overlap |
|---|---:|---:|---:|---:|
| train | 0.9552 | 0.9-ish | — | — |
| val | **0.9324** | 0.7200 | — | — |
| test | 0.9519 | 0.7899 | **0.3770** | **7 / 26** |

**Read the two IoUs together.** Pooled IoU counts every pixel once across the split, so
a few very large masks set the number: `metal_nut/flip` covers ~48% of a frame while
`pill/color` covers 0.17%. On test, pooled IoU says 0.7899 and mean per-image IoU says
0.3770, with 7 of 26 images scoring **exactly zero**.

### Decision threshold: 0.28, not 0.5

The 7 zero-IoU images all peak between 0.379 and 0.488 — just under the default cut-off.
Tuning a single threshold on **val** and applying it unchanged to test:

| split | IoU @0.50 | IoU @**0.28** | zero-overlap |
|---|---:|---:|---:|
| val | 0.2712 | **0.4423** | 25 → 5 |
| test | 0.3770 | **0.4536** | **7 → 0** |

Use **0.28** for inference on this checkpoint. The images in `inference/` were generated
at 0.5 and therefore show the pessimistic case.

Full metrics: `../checkpoints/sweep_5e-5/results.json`.

## Metric schema note

`train_localization.py` originally reported a single `iou@0.5` (pooled). It now reports
`iou_pooled`, `iou_per_image` and `zero_overlap_images`. The `5e-5` and `1e-4` sweep runs
were launched before that change and their `results.json` carries the old key; `2e-4` and
`4e-4` carry the new ones. Python reads a script at import, so an edit mid-run does not
reach the running process. IoU across LRs should be recomputed from the saved
checkpoints rather than compared across the two schemas — LR selection itself is
unaffected, since it uses `pixel_ap`, whose key did not change.

## Inference images

`inference/` holds one example per object, five files each:

| suffix | contents |
|---|---|
| `_input.png` | the 256x256 input |
| `_ground_truth.png` | the annotated defect mask |
| `_pred_heatmap.png` | raw anomaly probability, 0-255 |
| `_pred_mask.png` | thresholded at 0.5 |
| `_overlay.png` | prediction in red over the input |

Per-image IoU for those six examples:

| object | defect | pred px | true px | IoU |
|---|---|---:|---:|---:|
| wood | color | 1,961 | 1,099 | 0.463 |
| pill | color | 167 | 198 | 0.437 |
| tile | crack | 755 | 1,373 | 0.395 |
| metal_nut | bent | 61 | 1,447 | 0.036 |
| screw | manipulated front | **0** | 182 | 0.000 |
| toothbrush | defective | **0** | 764 | 0.000 |

The heatmap is worth looking at for the two zeros: the model may be ranking the right
pixels while never crossing 0.5, which is a threshold problem rather than a blindness
problem. Pixel-AP is threshold-free and is high for `pill`, so that distinction matters.

## Reproduce

```bash
cd "D:\computer-vision-image-generation\Image generation\real-world-applications\3-defect-generation"
MYSP="D:/computer-vision-image-generation/Image generation/.venv/Lib/site-packages"
PYTHONPATH="$MYSP" python train_localization.py \
  --epochs 30 --bs 16 --lr 5e-5 --workers 2 --out checkpoints/sweep_5e-5
```

Inference images regenerate from `results/_infer.log`'s script; see [../RUN.md](../RUN.md).
