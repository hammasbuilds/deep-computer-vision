# Video instance segmentation: the masks are steady, the identities are not

A per-frame detector run over video produces something that looks like video segmentation. The
usual way it is shown is a strip of frames with coloured masks, which hides the only question
that makes it a *video* result: does the mask on an object stay the **same** mask from one frame
to the next?

This scores [Mask R-CNN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html)
(`MaskRCNN_ResNet50_FPN_Weights.COCO_V1`, as published, no fine-tuning) frame by frame on the
**official DAVIS 2017 val split**, and reports per-frame quality, coverage and temporal
behaviour together — because any one of the three on its own is misleading. Run on Kaggle's GPU.

| | |
|---|---|
| split | official `ImageSets/2017/val.txt` — 30 sequences |
| frames scored | 1999 |
| ground-truth objects | 61, across 3929 instance-frames |
| predicted masks | 17632 |
| model | COCO-pretrained, never trained on DAVIS; published COCO mask AP 34.6 |

## Result

| metric | value |
|---|---|
| **matched-object mean IoU** (covered objects) | **0.76111** |
| **coverage** (objects that got any COCO-class detection) | **0.80328** — 49 of 61 |
| mean IoU over all 61 GT objects | 0.65242 |
| mean IoU over all GT instance-frames | 0.66446 |
| J mean (mean over sequences) | 0.72139 |
| instance-frames with IoU ≥ 0.5 | 0.77017 |
| class-agnostic mask AP / AP50 / AP75 | 0.22609 / 0.39496 / 0.24354 |

| temporal | value |
|---|---|
| **identity jump rate** | **0.10342** |
| presence flicker rate | 0.06111 |
| predicted mask frame-to-frame IoU | 0.78932 |
| GT mask frame-to-frame IoU (the motion baseline) | 0.78441 |
| stability gap (GT − predicted) | **−0.00491** |

## Finding 1: the model's video problem is association, not segmentation

The two temporal numbers disagree, and the disagreement is the result.

Frame-to-frame mask IoU for the predictions is **0.78932**. For the *ground-truth* masks of the
same objects over the same frame pairs it is **0.78441**. The gap is **−0.00491** — predicted
masks change very slightly *less* between consecutive frames than the true masks do. By that
measure the output is not jittery at all.

Yet on **0.10342** of consecutive frame pairs where the object is detected in both frames, the
mask covering it is not linked to the mask that covered it a frame earlier. The shape is stable;
the correspondence is what breaks.

This is exactly why the GT baseline has to be measured alongside. Reporting predicted
frame-to-frame IoU alone (0.78932) would read as "temporally consistent". Reporting the jump
rate alone would read as "jittery masks". Neither is true, and the fix each would suggest is
different: what this needs is a tracking component, not a smoother mask head.

The GT number is also what makes the predicted one interpretable. Without it, 0.78932 could mean
a stable model or a slow-moving dataset — DAVIS objects move enough that the true masks overlap
themselves by only 0.78441 between frames.

## Finding 2: coverage separates class mismatch from bad segmentation

Mean IoU over all 61 objects is **0.65242**. That single number is a blend of two unrelated
things, and splitting it changes the conclusion:

| | mean IoU |
|---|---|
| the 49 **covered** objects | **0.76111** |
| the **uncovered** objects | **0.20864** |

The uncovered objects, with their mean IoU:

| sequence | instance | mean IoU |
|---|---|---|
| paragliding-launch | 3 | 0.00297 |
| shooting | 3 | 0.07081 |
| shooting | 1 | 0.08752 |
| kite-surf | 1 | 0.12367 |
| gold-fish | 5 | 0.13152 |
| lab-coat | 1 | 0.18894 |
| lab-coat | 2 | 0.20497 |
| kite-surf | 2 | 0.24468 |
| loading | 2 | 0.28458 |
| paragliding-launch | 2 | 0.33668 |
| bmx-trees | 1 | 0.34593 |
| gold-fish | 3 | 0.48141 |

Read the list rather than the number: a goldfish, a paraglider canopy, lab coats as objects
distinct from the people wearing them, a firearm. **COCO has no category for any of them.** A
COCO-trained detector cannot find them, and that is honest class mismatch, not poor
segmentation. The things it *can* name it segments at 0.76111.

What it actually assigns to the DAVIS objects it matches well:

| COCO label | well-matched instance-frames |
|---|---|
| person | 1439 |
| dog | 279 |
| cow | 265 |
| bird | 263 |
| car | 215 |
| horse | 171 |
| motorcycle | 151 |
| skateboard | 49 |
| bicycle | 40 |
| sheep | 36 |

Reporting 0.65242 alone would invite the conclusion "Mask R-CNN segments video objects
moderately well". Reporting 0.76111 alone would hide that it never found a fifth of them.

## Why mask AP (0.22609) sits so far below mean IoU

Not a contradiction — a different question.

DAVIS annotates a handful of salient objects per sequence. Mask R-CNN predicted **17632** masks
against **3929** annotated instance-frames: it finds bystanders, vehicles and scenery that DAVIS
does not label. Under AP every one of those unannotated-but-real detections is a false positive,
so precision collapses. Mean IoU asks the opposite question — for each GT object, how well is it
covered — and is unaffected by extra detections.

AP here is largely a statement about DAVIS's annotation sparsity. Coverage and matched IoU are
the numbers to read.

Note also `fraction_instance_frames_any_overlap` = **0.98371**: almost every GT instance-frame
has *some* predicted mask touching it. Failure is rarely "nothing there"; it is "the wrong
extent, or the wrong identity".

## Where it fails

| sequence | J | identity jump rate |
|---|---|---|
| shooting | 0.2745 | 0.14286 |
| paragliding-launch | 0.3101 | 0.12308 |
| kite-surf | 0.3392 | 0.48276 |
| blackswan | 0.8719 | 0.12245 |
| parkour | 0.8821 | 0.0303 |

`kite-surf` is the worst case for identity: **0.48276** of its consecutive pairs jump. A small,
thin, fast-moving kite surfer is precisely the geometry that defeats IoU-based association.
`blackswan` shows the two axes are independent — J of 0.8719 with a jump rate of 0.12245.

## Input / Output

Input: DAVIS 2017 480p (`lennelenne/davis2017`), attached by reference on Kaggle. Nothing is
uploaded.
Output: `results.json` — per-frame quality, coverage with the full uncovered-object list, mask
AP, the temporal block with its definitions, and per-sequence breakdowns of both.

Notebook: [davis mask ap and temporal consistency](https://www.kaggle.com/code/muhammadhammas13/davis-mask-ap-and-temporal-consistency)

## Definitions, because these metrics have no single standard

- **identity jump rate** — over consecutive frame pairs where the GT object is detected
  (IoU ≥ 0.5) in both frames, the fraction where the predicted mask covering it at *t* and the
  one covering it at *t+1* are not matched to each other by Hungarian mask-IoU assignment at
  IoU ≥ 0.5. Deliberately not index-based: reordering the detection list is not a jump, which
  was verified against synthetic masks before the run.
- **presence flicker rate** — fraction of consecutive GT-object pairs detected in one frame but
  not the other.
- **coverage** — fraction of GT objects whose mean best-IoU over their frames reaches 0.5.
- **matched-object mean IoU** — mean of those per-object IoUs over covered objects only. It is
  computed per object, not per frame, so it cannot be inflated by conditioning each frame on
  having succeeded.
- **J** — per GT object, mean over its frames of its best predicted-mask IoU; averaged over
  objects, then over sequences.

## What this does not show

- **It is not a comparison.** One model on one dataset ranks nothing against anything.
- Mask R-CNN has no tracking component. These temporal numbers are the per-frame baseline a
  video method has to beat, not a criticism of a model doing a job it was not built for.
- Matched-object IoU excludes everything the model never found, and coverage is the companion
  number that says how much that is. Neither should be quoted alone; the all-objects figure of
  0.65242 is reported for that reason.
- Predicted frame-to-frame IoU is computed only on pairs detected in both frames, so it is
  conditioned on success. The flicker rate of 0.06111 is the companion number for the misses.
- Everything is at a fixed score threshold of 0.5, and the jump rate is a property of this
  specific linking rule. A different threshold moves all of these together.
