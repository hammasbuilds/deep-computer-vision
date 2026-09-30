# Product photo matting

Alpha matte extraction for background removal.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | U2-Net, 44.01 M parameters, fine-tuned |
| data | 2,674 product and person images at 320x320, derived alpha mattes |

## Result

| split | IoU / MAE |
|---|---|
| train | 0.9903 / 0.0037 |
| val | 0.9625 / 0.0099 |
| test | 0.9611 / 0.0104 |

## Notes

- U2-Net fine-tuned for 100 epochs with BCE plus soft-dice at 320x320.
- Ground-truth mattes are derived by Otsu thresholding with hole filling.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
