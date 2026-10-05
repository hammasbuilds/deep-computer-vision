# Industrial anomaly detection, without training

MVTec AD is fifteen categories of manufactured parts. The training set contains **only
defect-free** examples; the test set contains both. The usual approach trains a model per
category.

**Nothing here is trained.** A frozen ImageNet ResNet-50 produces patch features, the
features of the defect-free training images go into a memory bank, and a test patch is
scored by its distance to the nearest normal patch. The only fitted object is a record of
what normal looks like. CPU only.

## Result

| | mean image AUROC |
|---|---|
| **frozen ResNet-50 features** | **0.9379** |
| raw 32×32 pixels, identical pipeline | 0.6700 |
| published PatchCore (trains a coreset) | ~0.991 |

Bar was 0.90. Three categories separate perfectly: **bottle, hazelnut and leather all
reach 1.0000**, on 83, 110 and 124 test images respectively — large enough that this is
genuine separation rather than a small-sample artefact.

## The finding: what the features are worth depends on what you are inspecting

The raw-pixel control runs the same bank, the same distance and the same scoring rule —
only the representation differs. Splitting the fifteen categories into MVTec's own
texture and object groups:

| | features | raw pixels | the features are worth |
|---|---|---|---|
| texture (carpet, grid, leather, tile, wood) | 0.9634 | **0.8179** | +0.1455 |
| object (the other ten) | 0.9252 | **0.5960** | **+0.3292** |

**The representation is worth 2.3× more on objects than on textures.** On a texture, a
defect is a local statistical break and raw pixels already catch most of it — `leather`
scores **0.9808** on pixels alone against 1.0000 with features. On an object, the defect
is a part being wrong in a specific place, and raw pixels are near chance: `metal_nut`
scores **0.4040** on pixels — *below* 0.5, so worse than guessing — against **0.9878**
with features.

Without the control, 0.9379 would have read as a fact about the method. It is mostly a
fact about where the representation is doing the work.

## Per category

| category | features | raw pixels | test images | kind |
|---|---|---|---|---|
| bottle | 1.0000 | 0.6746 | 83 | object |
| hazelnut | 1.0000 | 0.7164 | 110 | object |
| leather | 1.0000 | 0.9808 | 124 | texture |
| carpet | 0.9976 | 0.5096 | 117 | texture |
| tile | 0.9964 | 0.8606 | 117 | texture |
| metal_nut | 0.9878 | 0.4040 | 115 | object |
| wood | 0.9860 | 0.9447 | 79 | texture |
| transistor | 0.9733 | 0.5404 | 100 | object |
| zipper | 0.9514 | 0.4999 | 151 | object |
| cable | 0.9413 | 0.5805 | 150 | object |
| pill | 0.9258 | 0.6721 | 167 | object |
| toothbrush | 0.9028 | 0.8486 | 42 | object |
| capsule | 0.8803 | 0.5297 | 132 | object |
| grid | 0.8371 | 0.7937 | 78 | texture |
| **screw** | **0.6893** | 0.4943 | 160 | object |

`screw` is the floor at 0.6893, and it is the category where a defect is a fine scratch
on an already-textured metal surface photographed at varying rotations — the one case
where "distance to the nearest normal patch" has the least to work with.

## Input / Output

Input: `ipythonx/mvtec-ad` on Kaggle, attached by reference. Nothing is uploaded.
Output: `results.json` — per-category AUROC for both representations, test counts, and
the bank cap.

Notebook: [MVTec anomaly without training](https://www.kaggle.com/code/muhammadhammas13/mvtec-anomaly-without-training)

## What this does not show

- **Not that training is unnecessary.** Published methods that train, and that build the
  memory bank with a proper coreset rather than a random subsample, score higher — 0.991
  against 0.9379 here.
- **Image-level only.** AUROC says a defect was detected, not that it was *located*.
  MVTec ships pixel masks; this notebook does not score against them, so nothing here
  speaks to localisation.
- The bank is capped at 8,000 randomly chosen patches per category for flat memory. A
  coreset of the same size would be a better bank, and that gap is part of the 0.05
  difference from the published figure.
- One backbone. A different pretrained network would move every number, and the
  texture/object split is the kind of result that could be backbone-specific.
