# deep-computer-vision

Recognition and understanding: segmentation, localisation and detection.

*A demonstration / portfolio project.*

| project | what | test result |
|---|---|---|
| [defect-localisation](defect-localisation/) | Industrial defect localisation | 0.9583 |
| [product-matting](product-matting/) | Product photo matting | 0.9611 / 0.0104 |
| [gdpr-anonymisation](gdpr-anonymisation/) | Face anonymisation | see folder |
| [pose-estimation](pose-estimation/) | 2D human keypoints on COCO val2017 | OKS AP 0.656, PCK@0.2 0.95364 |
| [video-action-recognition](video-action-recognition/) | UCF101 clip classification, official split 1 | top-1 0.80941, top-5 0.96405 |
| [video-instance-segmentation](video-instance-segmentation/) | DAVIS 2017 val, per-frame masks over video | matched IoU 0.76111 at coverage 0.80328 |
| [anomaly-detection](anomaly-detection/) | UCF-Crime surveillance benchmark | frame-level AUC 0.76329 |

## Method

The first three projects train a model on a fixed 80/15/5 split and report on train, validation
and test.

The four video and keypoint projects measure **published weights** on **official held-out
splits** instead — COCO val2017, UCF101 split 1, DAVIS 2017 val, the UCF-Crime test videos —
because what is being contributed there is the measurement rather than the training. Each one
states plainly what its number does and does not establish, and each carries the `results.json`
its Kaggle run produced. Every figure quoted in a project README is checked to appear in that
project's `results.json`.

Trained weights are not included: checkpoints run 109-294 MB against GitHub's 100 MB limit. Each
project carries its full metric history and the code to reproduce the run.
