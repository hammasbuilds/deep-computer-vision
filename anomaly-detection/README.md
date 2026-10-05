# Surveillance anomaly detection: a passing AUC and an unusable detector

**This is a benchmark measurement, not a theft detector.** It is not a detector of anything.
It produces a per-frame anomaly score, and the headline number says that score ranks anomalous
frames above normal ones better than chance. The operating-point table below says what it would
cost to actually use it, and that is the number worth reading: **to catch 80% of anomalous
frames it flags 43.754% of all frames.** Nothing here should be pointed at a real person or
place.

The method is weakly-supervised multiple-instance learning in the style of Sultani et al.:
training sees only **video-level** labels, and the reported number is frame-level ROC AUC on
held-out test videos against the **official UCF-Crime per-frame temporal annotations**.

| | |
|---|---|
| features | frozen ImageNet ResNet-50 (`IMAGENET1K_V2`), 2048-d, nothing fine-tuned |
| supervision at train | video-level only — "this video contains an anomaly" |
| labels at test | official per-frame temporal annotations, fetched at run time |
| annotation coverage | every official test video matched an annotation, and the folder label agreed with the annotation on every one of them (from the run log) |
| train / test videos | 1582 / 283, zero video-name overlap |
| test frames scored | 111134, of which 8520 anomalous (7.666%) |
| frame size | 64 × 64 — this dataset copy is decimated and small |

## Result

| metric | value |
|---|---|
| **frame-level AUC** | **0.76329** |
| frame-level AUC, per-frame scoring | 0.7574 |
| naive full-video-label baseline | 0.73574 |
| MIL advantage over naive | +0.02755 |

Model selection used video-level AUC on train videos held out from the fit (**0.92064** at
iteration 3000). Test frames were never used to select anything.

## The finding: AUC says it works, the operating points say it cannot be used

| required recall | false-positive rate | threshold |
|---|---|---|
| 0.50023 | **0.1543** | 0.37777 |
| 0.8 | **0.43754** | 0.07476 |
| 0.90035 | **0.53227** | 0.03977 |

Anomalous frames are **7.666%** of the test set. At the operating point that catches 80% of
them, 43.754% of all frames are flagged — so the overwhelming majority of alarms are on normal
footage, and raising recall to 0.90035 pushes false positives past half of everything.

A single AUC of 0.76329 reads as a working system. It is the same model at every threshold, and
there is no threshold at which it is usable. That gap is the entire reason this write-up leads
with the caveat rather than the metric: AUC is a ranking summary over a pooled frame set, and
pooled ranking quality is not operational performance.

## The weak-supervision machinery buys less than expected

The naive thing to do with video-level labels is to pretend every frame of an anomalous video
is anomalous and fit a classifier. That is wrong — most frames of an anomalous video are
normal — and MIL exists to avoid it.

Under identical features, identical splits and the same 32-segment representation, the naive
classifier scores **0.73574** and MIL scores **0.76329**: an advantage of **+0.02755**. Real,
reproducible, and much smaller than the framing of the method suggests. On these features, most
of the available signal is apparently recoverable without the ranking loss at all.

That comparison is the one controlled contrast here, because both sides were trained in this
notebook on the same inputs.

## Per-class frame AUC

Each class's anomalous frames scored against all normal-video frames.

| class | AUC |
|---|---|
| Arrest | 0.6761 |
| Robbery | 0.73109 |
| Fighting | 0.75746 |
| RoadAccidents | 0.7656 |
| Shoplifting | 0.81154 |
| Explosion | 0.85644 |
| Shooting | 0.85734 |
| Burglary | 0.8931 |
| Vandalism | 0.90262 |
| Stealing | 0.90544 |
| Arson | 0.92937 |
| Assault | 0.95166 |

The theft-related classes the dataset is often fetched for do not behave alike: Stealing
(0.90544) and Burglary (0.8931) score well above Shoplifting (0.81154) and Robbery (0.73109).
Arrest is worst at 0.6761 — unsurprising, since an arrest looks like several people standing
near a vehicle, which is also what a great deal of normal footage looks like.

These are the dataset's own class folders. They say nothing about intent, legality or identity.

## Input / Output

Input: UCF-Crime extracted frames (`odins0n/ucf-crime-dataset`), attached by reference on
Kaggle. Nothing is uploaded. The official temporal annotations are fetched at run time; the run
asserts rather than fabricating frame labels from video labels if they cannot be obtained.
Output: `results.json` — the AUC under both scoring protocols, the operating-point table,
per-class AUCs, the naive baseline, the selection history and the frame counts.

Notebook: [ucf crime frame level auc](https://www.kaggle.com/code/muhammadhammas13/ucf-crime-frame-level-auc)

## This is not a deployable system

- It must not be pointed at a real person or place. It detects nothing; it ranks frames.
- UCF-Crime's videos were selected and trimmed by annotators. Its "normal" videos are not a
  sample of real surveillance footage, so this AUC does not transfer to a live camera.
- An anomaly score is not a crime classification.
- No face, person or identity is detected, tracked or recognised anywhere in this notebook.

## What this does not show

- One feature extractor and one MIL head on one benchmark. It establishes no ranking against
  published UCF-Crime methods, which use C3D or I3D features at full frame rate.
- The frames in this Kaggle copy are 64 × 64 and decimated, so this number is not comparable to
  papers that use every frame at full resolution. That it lands near the published figure
  anyway is interesting but not evidence of equivalence.
- Test videos with too few frames to form 32 segments were dropped, leaving the 283 scored
  here out of the official test set.
- The MIL-versus-naive gap is a controlled contrast. The absolute AUC is not.
