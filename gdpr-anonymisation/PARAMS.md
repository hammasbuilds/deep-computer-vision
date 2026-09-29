# Run parameters — GDPR face anonymisation

Inference only. Nothing is trained, so there are no training hyperparameters — the
settings below are what produced the images in `inference/`.

## Model

| | |
|---|---|
| model | CenterFace |
| weights | shipped inside the `deface` package — **nothing to download** |
| runtime | `onnxruntime`, dynamic input shape |
| library | `deface` (installed from the cloned repo) |

`onnx` is a hard requirement, not optional: `CenterFace(in_shape=None, ...)` rewrites the
ONNX graph for dynamic shapes and does `import onnx` at construction. Without it the run
dies **after** loading the image, which reads like a data problem rather than a missing
package.

## Settings

| setting | value |
|---|---|
| `in_shape` | `None` (dynamic, native resolution) |
| backend | `onnxrt` |
| detection threshold | 0.2 |
| masking | solid black box over each detection |

### Why threshold 0.2

For anonymisation a **low** threshold is the safe default: a missed face is a privacy
failure, an extra box is a cosmetic one. The trade-off is visible in the sweep below.

## Results

Input `deface/examples/city.jpg`, 800 x 564:

| | |
|---|---|
| detection time | **0.40s** (CPU) |
| boxes at 0.2 | 25 |
| regions masked | 25 |
| pixels changed | 14,626 (**3.24%** of frame) |

### Threshold sweep

| threshold | faces |
|---:|---:|
| 0.2 | 25 |
| 0.4 | 20 |
| 0.6 | 16 |
| 0.8 | 6 |

Always report the curve, not a single count. A smooth falloff means the confidence
scores carry information; a flat curve would mean they do not and the count is noise.

## Inference images

| file | contents |
|---|---|
| `city_input.jpg` | original |
| `city_anonymised.jpg` | 25 regions masked at threshold 0.2 |

## Fine-tuning

Not done. CenterFace is used as shipped. Fine-tuning needs images with face bounding
boxes — WIDER FACE (~1.5 GB) or FDDB (~600 MB), neither on this drive. See
[../../../DATASETS.md](../../../DATASETS.md).

## `deep_privacy2`

Also cloned here (219 files). It *replaces* faces with generated ones rather than masking
them, needs a GPU, and downloads its own checkpoints. `deface` is the one that runs today
on CPU with zero downloads; `deep_privacy2` is the higher-quality option once its weights
are fetched.

## Reproduce

```bash
cd "D:\computer-vision-image-generation\Image generation"
.venv/Scripts/python.exe - <<'PY'
import numpy as np, imageio.v2 as iio
from deface.centerface import CenterFace
img = iio.imread("real-world-applications/2-gdpr-anonymisation/deface/examples/city.jpg")
dets, _ = CenterFace(in_shape=None, backend="onnxrt")(img, threshold=0.2)
out = img.copy()
for x1, y1, x2, y2, s in dets:
    if s < 0.2: continue
    x1, y1, x2, y2 = (int(max(0, v)) for v in (x1, y1, x2, y2))
    out[y1:y2, x1:x2] = 0
iio.imwrite("out.jpg", out)
PY
```

Full commands: [../RUN.md](../RUN.md).
