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

- Trained from scratch for 120 epochs across 12 MVTec AD object categories.
- Test pixel AP 0.9583, pooled IoU 0.8033, per-image IoU 0.6232. The decision threshold was tuned on the validation split.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
