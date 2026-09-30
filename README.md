# deep-computer-vision

Recognition and understanding: segmentation, localisation and detection.

*A demonstration / portfolio project.*

| project | what | test result |
|---|---|---|
| [defect-localisation](defect-localisation/) | Industrial defect localisation | 0.9583 |
| [product-matting](product-matting/) | Product photo matting | 0.9611 / 0.0104 |
| [gdpr-anonymisation](gdpr-anonymisation/) | Face anonymisation | see folder |

## Method

Every model is trained on a fixed 80/15/5 split and reported on train, validation and test.

Trained weights are not included: checkpoints run 109-294 MB against GitHub's 100 MB limit. Each project carries its full metric history and the code to reproduce the run.
