# UCF101 action recognition: the mistakes are all inside the right scene

A linear probe on frozen Kinetics-400 features gets **0.80941** top-1 on the official UCF101
split-1 test set. The interesting part is not the number, it is that **every leading error is a
confusion between two actions that share a scene, a sport or an instrument family** — swimming
stroke for swimming stroke, drum for drum, cricket shot for cricket bowl. The features know
where they are and not quite what is happening there.

[`r3d_18`](https://pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html)
pretrained on Kinetics-400 is used **frozen**, as a clip feature extractor. A multinomial
logistic regression is fitted on the train split only, `C` is chosen on val, and the reported
number is the test split, which the classifier never sees.

| | |
|---|---|
| split | **official UCF101 split 1** (`trainlist01`/`testlist01`), fetched at run time and validated against the published split-1 list sizes before use |
| classes | 101 |
| videos used | 5050 train / 1082 val / **3783 test** |
| leakage checks | file overlap 0 and **clip-group overlap 0** on all three split pairs |
| decode success | 1.0 on all three splits |
| device | cpu — the Kaggle weekly GPU quota was exhausted |

## Result

| metric | value |
|---|---|
| **test top-1** | **0.80941** |
| **test top-5** | **0.96405** |
| val top-1 (used only to pick `C`) | 0.8512 |
| train top-1 | 1.0 — fit quality, not a result |

`C` sweep on val: 0.1 → 0.7634, 1.0 → 0.84658, 10.0 → 0.84473, 100.0 → 0.8512. Selected
**100.0**, on val, never on test.

The probe reaches **1.0** on its own training videos, so the test number is not limited by
classifier capacity. The ceiling is in the frozen features.

## The finding: the confusion pairs are all within-domain

| n | rate of true class | true | predicted |
|---|---|---|---|
| 17 | 0.459 | FrontCrawl | BreastStroke |
| 14 | 0.286 | CricketShot | CricketBowling |
| 13 | 0.361 | BrushingTeeth | ShavingBeard |
| 13 | 0.265 | PlayingDhol | PlayingTabla |
| 12 | 0.333 | BrushingTeeth | ApplyLipstick |
| 11 | 0.306 | YoYo | Nunchucks |
| 11 | 0.229 | PlayingFlute | PlayingViolin |
| 11 | 0.224 | BoxingPunchingBag | BoxingSpeedBag |
| 9 | 0.273 | MilitaryParade | BandMarching |
| 9 | 0.25 | Kayaking | Rafting |
| 9 | 0.2 | HammerThrow | ThrowDiscus |
| 8 | 0.258 | BalanceBeam | ParallelBars |

Not one of these is a random mistake. Read as pairs they are almost tautological:

- **Same sport, different act** — CricketShot/CricketBowling, BoxingPunchingBag/BoxingSpeedBag,
  BalanceBeam/ParallelBars, HammerThrow/ThrowDiscus.
- **Same medium** — FrontCrawl/BreastStroke (a pool), Kayaking/Rafting (a river).
- **Same instrument family** — PlayingDhol/PlayingTabla (drums), PlayingFlute/PlayingViolin.
- **Same scene and posture** — BrushingTeeth confused with both ShavingBeard *and* ApplyLipstick:
  a hand raised to the face at a bathroom mirror, three times over.
- **Same object kinematics** — YoYo/Nunchucks, two things swung on a string.

The pattern says what the representation is doing. Kinetics features encode scene, object and
context strongly, and a linear read-out of them separates *contexts* almost perfectly — top-5 is
**0.96405**, so the right answer is nearly always among the handful of candidates. What a linear
probe cannot do is the fine-grained temporal discrimination *within* a context: which swimming
stroke, which cricket action, which bathroom gesture.

That is also why the worst classes are what they are:

| class | accuracy | n |
|---|---|---|
| BrushingTeeth | 0.25 | 36 |
| CricketBowling | 0.25 | 36 |
| YoYo | 0.2778 | 36 |
| CricketShot | 0.2857 | 49 |
| Nunchucks | 0.3143 | 35 |
| WallPushups | 0.3429 | 35 |
| JumpRope | 0.3684 | 38 |
| FrontCrawl | 0.4054 | 37 |

Every one of them has a near-twin elsewhere in the label set. The gap between top-1 **0.80941**
and top-5 **0.96405** is almost entirely this: the model is choosing wrongly among a small set of
correct-context neighbours.

## A group-disjoint validation split still overestimates the official test split

Val here is carved out of the official *train* list by clip group, so it shares no video and no
group with either the fit set or the test set. It still reads optimistically: val top-1
**0.8512** against test top-1 **0.80941**.

The val groups come from the same pool of recordings as the fit groups, and UCF101 recordings
within a class resemble each other even across groups. A disjoint split is necessary but not
sufficient; the official test partition is a genuinely different sample, and only it is quoted
as the result.

## Why the split is not the dataset's own folders

The Kaggle dataset ships `train/`, `val/` and `test/` directories. They are not used, because
they **shared 30 video files outright** — a first run asserted and stopped on exactly that. The
deeper problem is that they split by *clip*: UCF101 clips come in groups
(`v_YoYo_g25_c01 … c05`) cut from one recording, so a clip-level split puts near-duplicate clips
of the same scene on both sides and inflates accuracy.

This run pools every video from all three directories, skipping duplicate basenames,
applies the official split-1 lists, carves val out by group, and asserts **zero file overlap and
zero group overlap** on all three pairs before measuring anything.

## Input / Output

Input: UCF101 (`matthewjansen/ucf101-action-recognition`), attached by reference on Kaggle.
Nothing is uploaded. The official split lists are fetched at run time and validated against the
published sizes.
Output: `results.json` — top-1/top-5 on test, the val `C` sweep, per-class accuracy, the 20
leading confusion pairs, split provenance and the leakage-check results.

Notebook: [ucf101 top1 top5 and confusion pairs](https://www.kaggle.com/code/muhammadhammas13/ucf101-top1-top5-and-confusion-pairs)

## What this does not show

- Run on **cpu** with **1** clip per video and training capped at **50** videos per class,
  because the Kaggle weekly GPU quota was exhausted. The val and official test splits were not
  reduced, so the metric is over the whole official test set — but a three-clip run on the full
  train split would score higher, and this number should be read as a floor rather than this
  method's best.
- **This is a linear probe on frozen features, not a fine-tuned video model.** A fine-tuned
  `r3d_18` scores considerably higher, and this is not comparable to published UCF101
  fine-tuning results.
- Kinetics-400 and UCF101 overlap in content, so some of this accuracy is pretrained-label
  affinity rather than transfer.
- One backbone on one dataset ranks nothing against anything.
- The confusion-pair reading is an interpretation of which labels get swapped. It is consistent
  across every leading pair and with the top-1/top-5 gap, but it is not a causal probe of the
  representation.
