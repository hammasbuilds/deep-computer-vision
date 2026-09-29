# Face anonymisation

Detect and obscure faces in crowded scenes.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | Pretrained detector plus blurring - no training |
| data | WIDER FACE validation split, 3,226 images |

## Notes

- This project trains nothing. It applies a published detector and blurs what it finds, so there is no learning curve and no held-out metric - it is included because it works and is useful, not as a modelling result.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
