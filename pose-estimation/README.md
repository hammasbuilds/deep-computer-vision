# 2D human pose: which joint is worst depends entirely on how strict you are

Everyone knows wrists and ankles are the hard joints. On this measurement that is true at a
loose tolerance and **false at a tight one** — at PCK@0.05 the worst joint on the body is the
**hip**, by a wide margin, and wrists score better than ankles, knees and hips alike.

This scores [Keypoint R-CNN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html)
(`KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`, as published, no fine-tuning) on COCO
**val2017**, held out from the train2017 set those weights were fitted on. Run on Kaggle's GPU.

| | |
|---|---|
| images scored | 2346 — every image this dataset copy ships that has an annotated person |
| ground-truth persons | 6352, of which 6237 matched a detection at box IoU ≥ 0.5 |
| labelled joints scored | 67391 of 68215 |
| protocol | official COCO OKS AP via `pycocotools`, plus PCK at three tolerances |

## Result

| metric | value | published for these weights |
|---|---|---|
| **OKS AP** | **0.656** | 65.0 |
| OKS AP50 / AP75 | 0.86695 / 0.71911 | |
| OKS AP medium / large | 0.60705 / 0.73455 | |
| OKS AR | 0.71815 | |
| PCK@0.2 | **0.95364** | |
| PCK@0.1 | 0.90539 | |
| PCK@0.05 | 0.81584 | |

The measured OKS AP of **0.656** lands on the published **65.0** for these weights. That
agreement is the evidence that the protocol here is right; it is not a new result about the
model. The run was judged against a pass bar of OKS AP 0.60 set from that published figure,
on the rule that a score far below it would be a bug in this pipeline rather than a finding
about the model.

PCK figures above are over matched persons. Counting the 115 unmatched ground-truth persons as
a total miss on every labelled joint gives PCK@0.2 **0.94212**, PCK@0.1 **0.89445**, PCK@0.05
**0.80598** — the detector-inclusive numbers.

## The finding: the per-joint ranking inverts with tolerance

| joint | PCK@0.05 | PCK@0.2 | median error |
|---|---|---|---|
| left_hip | **0.6388** | 0.9418 | 0.03637 |
| right_hip | **0.6389** | 0.9438 | 0.03614 |
| left_ankle | 0.745 | **0.9192** | 0.02244 |
| right_ankle | 0.7496 | **0.918** | 0.02296 |
| left_knee | 0.7677 | 0.9337 | 0.02097 |
| right_knee | 0.779 | 0.9359 | 0.02134 |
| left_wrist | 0.7933 | 0.9268 | 0.01813 |
| right_wrist | 0.8089 | 0.9335 | 0.01803 |
| right_elbow | 0.8113 | 0.9545 | 0.01883 |
| left_elbow | 0.8167 | 0.9528 | 0.0192 |
| right_shoulder | 0.8187 | 0.9651 | 0.02151 |
| left_shoulder | 0.8189 | 0.9644 | 0.02125 |
| right_ear | 0.9389 | 0.9774 | 0.01071 |
| left_ear | 0.9447 | 0.9838 | 0.01034 |
| nose | 0.9591 | 0.9822 | 0.0088 |
| left_eye | 0.9603 | 0.9822 | 0.00783 |
| right_eye | 0.9607 | 0.9824 | 0.00799 |

Two different joints are "the worst" depending only on the threshold:

- At **PCK@0.2** the ankles are last (0.918, 0.9192) and the hips sit comfortably mid-table
  (0.9418, 0.9438).
- At **PCK@0.05** the hips are last by a distance (0.6388, 0.6389), while the ankles (0.745,
  0.7496) rank above the knees and below the wrists.

The median-error column explains why, and the two failure modes are genuinely different:

- **Hips carry large typical error but a light tail.** Median error 0.03637 is the worst on the
  body, against 0.00783 at the eyes. Yet its PCK@0.2 is 0.9418, so few hip predictions miss by
  more than that tolerance. A hip has no visible landmark; it is an interior anatomical guess, so almost every
  prediction is a little bit off and almost none is wildly off.
- **Ankles carry small typical error but a heavy tail.** Median error 0.02244 is well under the
  hips', yet its PCK@0.2 of 0.918 is the lowest on the body — the heaviest tail there is. An ankle is a sharp, visible
  landmark that is either found precisely or lost completely to occlusion and motion blur.

So "wrists and ankles are worst" describes a *tail*, and the hips' problem is a *bias*. A single
PCK number cannot distinguish those, and the one most often quoted — the loose one — reports
only the tail.

### The loose threshold also hides almost all per-joint structure

| | spread across the joints |
|---|---|
| PCK@0.05 | **0.6388 → 0.9607** |
| PCK@0.2 | 0.918 → 0.9838 |

At PCK@0.2 every joint scores between 0.918 and 0.9838. The table is nearly flat and reads as
"this model is uniformly good". The same predictions at PCK@0.05 run from 0.6388 to 0.9607 and
rank the joints in a different order. The per-joint table is only informative at a tolerance tight
enough to resolve it.

Grouped at PCK@0.2, extremities (wrists + ankles, 13698 joints) score 0.92524 against
0.95931 for the core (shoulders + hips + nose, 24477 joints) — a real but modest gap that
understates what is happening underneath.

## Input / Output

Input: COCO 2017 keypoints (`asad11914/coco-2017-keypoints`), attached by reference on Kaggle.
Nothing is uploaded.
Output: `results.json` — OKS AP block, all three PCK tolerances with full per-joint tables,
per-joint median error, and the match/normalisation rules.

Notebook: [pose pck and oks per joint](https://www.kaggle.com/code/muhammadhammas13/pose-pck-and-oks-per-joint)

## Definitions

- **PCK@a** — a joint is correct if `||pred − gt|| ≤ a · max(gt_box_w, gt_box_h)`. Only
  keypoints with COCO visibility flag `v > 0` are scored. The normaliser matters: the same
  predictions score differently under torso- or head-normalised variants, so it is stated rather
  than assumed.
- **median error** — median of `||pred − gt||` under that same normaliser, per joint.
- **matching** — greedy box-IoU at threshold 0.5, detections taken in score order.

## What this does not show

- **It is not a comparison.** One model on one dataset establishes nothing about how this model
  ranks against any other.
- The scores belong to the published weights. No training happened here; the contribution is the
  measurement.
- **This is a subset of val2017.** The dataset copy ships 2346 annotated-person images, not the
  full official person split, so the OKS AP is computed over those 2346 images. It agrees with
  the published full-split number of 65.0, but it is not literally the official figure.
- val2017 is held out from training, yet it is the benchmark this model family was developed
  against. It is not an out-of-distribution test.
- The per-joint inversion is a property of this model on this dataset under this normaliser. It
  is a caution about how PCK is reported, not a general law about hips.
