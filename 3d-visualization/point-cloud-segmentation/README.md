# Point-cloud part segmentation

PointNet trained **from scratch** — no pretrained weights — to label the parts of a 3D
shape: which points of a chair are the legs, which of an aeroplane are the wings. 2,048
points per shape, 16 categories, 50 part labels. Run on Kaggle's GPU.

| | |
|---|---|
| model | PointNet with the segmentation head, input and feature T-Nets, 3.649 M parameters |
| data | ShapeNetPart, official train/val/test split |
| training | 40 epochs, batch 32 |
| selection | the epoch is chosen on **validation** instance mIoU; test is untouched until the end |

## Result

| split | instance mIoU | class mIoU |
|---|---|---|
| train | 0.8583 | 0.8153 |
| val | 0.8516 | 0.7774 |
| **test** | **0.8254** | **0.7702** |

Train 0.8583 against test 0.8254 is a gap of 0.0329 — the model is fitting the task, not
the training set.

## The finding: the two mIoUs disagree because rare categories are hard

Instance mIoU averages over shapes, class mIoU averages over categories. They differ by
**0.0552** on test, and the reason is visible once categories are split by how many test
shapes they have:

| test categories | mean IoU |
|---|---|
| rare — fewer than 60 shapes (8 categories) | **0.7016** |
| common — 200 or more shapes (4 categories) | **0.8249** |

A gap of **0.1233**. Class mIoU weights a category with 12 shapes the same as one with
704, so it reports the rare-category difficulty that instance mIoU averages away.

Worst and best on test:

| category | IoU | test shapes |
|---|---|---|
| Rocket | 0.4929 | 12 |
| Motorbike | 0.5713 | 51 |
| Earphone | 0.6939 | 14 |
| … | | |
| Chair | 0.8906 | 704 |
| Guitar | 0.8961 | 159 |
| Laptop | 0.9547 | 83 |

Laptop at 0.9547 with only 83 shapes shows the driver is not sample count alone — a
laptop is two near-planar parts with an unambiguous boundary, while a rocket's fins,
body and nose meet at boundaries that are genuinely ambiguous. **Sample size and
intrinsic difficulty both contribute, and this experiment does not separate them.**

## Input / Output

Input: ShapeNetPart HDF5 (`horsek/shapenetpart-hdf5-2048`), attached by reference on
Kaggle. Nothing is uploaded.
Output: `results.json` — all three splits, per-category IoU and per-category shape counts.

Notebook: [PointNet part segmentation](https://www.kaggle.com/code/muhammadhammas13/pointnet-part-segmentation-iou)

## What this does not show

- **Not a model comparison.** One architecture on one dataset says nothing about whether
  PointNet beats DGCNN or Point Transformer.
- Published PointNet reports about 83.7% instance mIoU on this benchmark; 82.5% from
  scratch in 40 epochs is close but not a reproduction attempt, and the difference is not
  investigated here.
- The rare/common split above is descriptive. It does not establish that sample size
  *causes* the difficulty, and the Laptop result is a direct counterexample to that
  reading.

## A detail that would have broken the run

The HDF5 keys in this packaging are `data` / `label` / `seg`. Almost all published
ShapeNetPart code assumes `data` / `label` / `pid`, and hardcoding that would have failed
on the first cell. The loader identifies arrays by **shape** instead: 3-D with last
dimension ≥ 3 is coordinates, 2-D with second dimension > 3 is part labels, 2-D with
second dimension 1 is the category. The label-numbering check is an assertion rather than
a warning, because a silently misassigned per-category table is worse than no table.
