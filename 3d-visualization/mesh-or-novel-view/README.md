# Novel-view synthesis

A hash-grid NeRF, Instant-NGP's encoding reimplemented in plain PyTorch, trained from
scratch on 100 views of one scene and scored on **25 poses it never saw**. Run on
Kaggle's GPU.

| | |
|---|---|
| scene | NeRF-synthetic `chair`, 400×400 |
| model | multiresolution hash grid, 16.777 M table parameters + a 9,555-parameter MLP |
| training | 20,000 steps × 4,096 rays, 128 samples per ray |
| held out | `transforms_test.json` poses; no test view used in training |

## Result

| | PSNR | SSIM |
|---|---|---|
| **held-out test views (25)** | **28.235** | **0.9488** |
| training views (subset) | 33.592 | 0.9729 |

Spread across the 25 test views is wide: sd 3.587, from **18.933** to **34.242**.

## The finding: nearly all of the model is the lookup table, and it memorises

The MLP has **9,555** parameters. The hash table has **16.78 million** — 1,756× more. So
this is not a network that learned a scene so much as a lookup structure that stored one,
with a tiny decoder on top.

The train/test gap shows what that costs: **33.592 dB on training views against 28.235 on
held-out ones, a gap of 5.357 dB**. Rendering from a pose it was fitted on is markedly
easier than rendering from one it was not, which is exactly what a memorising
representation predicts and what a single training-view render would have concealed.

The per-view spread matters as much as the mean. The worst held-out view scores
**18.933** against a best of 34.242 — a range of 15.3 dB on the same trained model. A
single averaged PSNR hides that some viewpoints are reconstructed far worse than others.

## Input / Output

Input: `nguyenhung1903/nerf-synthetic-dataset`, attached by reference on Kaggle.
Nothing is uploaded.
Output: `results.json` — both splits, per-view PSNR for all 25 test views, and the
batch-reduction log.

Notebook: [NeRF novel-view PSNR](https://www.kaggle.com/code/muhammadhammas13/nerf-novel-view-psnr)

## What this does not show

- **One scene.** `chair` is a clean synthetic object on a blank background. Nothing here
  speaks to real captures, unbounded scenes, or imperfect camera poses.
- **Not a reproduction.** A full Instant-NGP with fused CUDA kernels reaches well above
  this on the same scene; a plain-PyTorch implementation at 20,000 steps is a smaller
  experiment and is not a statement about the method's ceiling.
- SSIM is computed on the same renders as PSNR and inherits the same per-view spread.

## A cost worth recording

The run took **over eight hours** against a 40–50 minute estimate. I attributed that at
the time to the out-of-memory fallback halving the ray count — the notebook records
`oom_batch_reductions: []`, so **there were none**: it ran all 20,000 steps at the full
4,096 rays. The cost is simply that 20,000 × 4,096 rays × 128 samples is expensive
without fused kernels, and my estimate was wrong by roughly 8×.

That single run consumed most of a 30-hour weekly GPU budget. The lesson is not about
the fallback but about the estimate: a job whose cost is not measured on a small run
first can quietly spend a week's compute.
