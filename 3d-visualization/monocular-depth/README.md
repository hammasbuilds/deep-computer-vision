# Monocular depth, measured against ground truth

A single photograph does not contain depth. A monocular model predicts it anyway, and the
usual way this is shown is a colourful depth map beside the photo — which looks convincing
and proves nothing.

This scores [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
as published, no fine-tuning, against NYU Depth V2's **Kinect-measured** depth on the
official held-out test split. Run on Kaggle's GPU.

| | |
|---|---|
| images | 654 — the complete official NYU v2 test split |
| ground truth | measured depth, uint16 millimetres |
| evaluation range | 0.01–10.0 m, the NYU convention |
| raw model output | disparity — **determined empirically**, Spearman +0.9052 against ground-truth disparity (from the run log), not assumed |

## Result

| metric | value | spread across images (sd) |
|---|---|---|
| abs_rel | **0.09518** | 0.04245 |
| RMSE | **0.43383 m** | 0.27372 |
| log10 | 0.04331 | 0.01993 |
| δ < 1.25 | **91.137%** | 9.047 |
| δ < 1.25² | 97.283% | 3.412 |
| δ < 1.25³ | 98.845% | 1.499 |

## The finding: the alignment protocol moves the score more than the model does

Depth Anything V2 is a **relative** depth model — affine-invariant *in disparity*, meaning
it recovers depth ordering up to an unknown scale **and shift**. The protocol most
write-ups reach for, median-scaling in depth space, fits only the scale and leaves the
shift unmodelled.

Both columns below score **the same predictions**:

| metric | affine fit (scale + shift, correct) | median-scale only |
|---|---|---|
| abs_rel | 0.09518 | 0.36980 |
| RMSE | 0.43383 | 1.79099 |
| δ < 1.25 | 0.91137 | 0.46082 |

**δ<1.25 moves from 46.1% to 91.1% on identical predictions.** Reported under the wrong
protocol, this model looks roughly half as good as it is, and the number would read as a
fact about the model rather than about the measurement.

The fitted disparity transform is reported rather than hidden, because it is the thing a
single image cannot determine: scale **0.0848** (sd 0.03078), shift **0.25174** (sd 0.09518).

That shift standard deviation, 0.09518, happens to equal `abs_rel` to five
decimals. It is a coincidence, not a copied field: the run computes and prints
the fit statistics on a separate path from the metrics, and the log reports
them independently.

## Where the error lives

| region | abs_rel |
|---|---|
| near a depth edge | 0.13158 |
| away from edges | 0.08635 |

Error is **1.52× higher** at depth discontinuities. A mean over the whole image hides
this, and depth edges are exactly where a predicted depth map is used — for occlusion,
for segmentation boundaries, for anything that back-projects to geometry.

## Input / Output

Input: NYU Depth V2 (`soumikrakshit/nyu-depth-v2`), attached by reference on Kaggle.
Nothing is uploaded.
Output: `results.json` — both protocols, the per-image spread, the fitted transform and
the edge split.

Notebook: [Monocular depth vs ground truth](https://www.kaggle.com/code/muhammadhammas13/monocular-depth-vs-ground-truth)

## What this does not show

- **It is not a comparison.** One model on one dataset establishes nothing about whether
  this model beats another.
- The scores are the published model's, not a result of training done here. What this
  contributes is the measurement.
- RMSE depends on the depth-PNG decoding; `abs_rel`, `log10` and the deltas do not. The
  decoding was detected by testing which candidate yields a plausible indoor range, and
  the choice is printed in the run.
- 654 indoor scenes from one sensor. Nothing here speaks to outdoor scenes or other depth
  sensors.

## Two bugs this went through, both worth recording

**The first version finished `COMPLETE` and scored zero images.** Its RGB/depth classifier
tested the whole path for `"depth"`, and the dataset mounts at `/kaggle/input/nyu-depth-v2/`
— so all 102,684 files were classified as depth maps and nothing could pair. The discovery
cell that should have exposed this pruned its walk one level above where the images live
and printed "0 directories with files". A green run that measured nothing. Pairing now
asserts on fewer than 20 pairs, so a run that cannot measure fails loudly.

**The second version produced `abs_rel` of 95,534.** Unbounded `1/disparity` sent
far-plane pixels to 10⁶ m. The giveaway was that `log10` and `delta1` stayed sane while
the mean-of-ratio metrics exploded — the signature of a thin tail of huge outliers rather
than a broadly wrong prediction.
