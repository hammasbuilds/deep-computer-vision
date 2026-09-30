# Face anonymisation

Detect and obscure faces in crowded scenes.

*A demonstration / portfolio project.*

| | |
|---|---|
| model | Pretrained detector plus blurring - no training |
| data | WIDER FACE validation split, 3,226 images |

## Notes

- Applies a published face detector and blurs the detected regions. No training is involved; the samples show the pipeline on WIDER FACE.

## Contents

| path | what |
|---|---|
| `results.json` | every split and the per-epoch history |
| `samples/` | input, output and ground truth |
| `*.py` | the training and data-preparation code |

Trained weights are not in the repo: the checkpoints run 109-294 MB and GitHub rejects files over 100 MB. `results.json` carries the full history and the code reproduces the run.
