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

- MAE moved 0.4464 -> 0.0104 when a double-activation bug was found. U2-Net applies a sigmoid inside its own forward pass and the training code applied another, clamping every output to the range [0.500, 0.731]: the model could not express a confident prediction at all. IoU hid it completely, because thresholding at 0.5 sits exactly at the squashed background level, so IoU read a healthy 0.82 while the mattes were two shades of grey.
- The ground-truth mattes are derived by Otsu thresholding with hole filling, not hand-labelled, so the ceiling here is the quality of that derivation.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
