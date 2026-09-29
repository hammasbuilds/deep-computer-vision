# Industrial defect localisation

Pixel-level localisation of manufacturing defects.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | DiscriminativeSubNetwork, trained from scratch |
| data | MVTec AD, 12 object categories, split 1003/190/66 images |

## Result

| split | pixel AP |
|---|---|
| train | 0.9839 |
| val | 0.9565 |
| test | 0.9583 |

For comparison:

| | |
|---|---|
| at epoch 0 | 0.155 |

## Notes

- Both IoU figures are reported because they disagree and the difference matters: pooled IoU on test is 0.8033 but per-image IoU is 0.6232. Pooled IoU sums intersections and unions across the set, so large defects dominate it. Quoting the pooled figure alone would flatter the result by about 0.18.
- Test 0.9583 against val 0.9565 - it is not overfitted to the validation split.
- 5 of 66 test images have zero overlap with the ground-truth mask. The decision threshold was tuned on validation (0.28, not 0.5), which took per-image IoU from 0.3770 to 0.4536 and zero-overlap images from 7 to 0 on an earlier run.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
