# Gradient Influence Analysis

This directory contains script for Gradient Influence Analysis.

## Method

For each sample, the script computes the gradient of a scalar target from the final logits with respect to the input embeddings:

- Input embeddings: `E in R^{T x d_model}`.
- Target scalar: by default, the summed logits averaged over the last `N_tau = 20` positions, matching the paper.
- Per-position influence score: `|| d target / d E_i ||_2`.

Larger scores indicate stronger influence on the final prediction.

## Usage

```bash
python run_gradient_influence.py \
  --dataset-path DATASET_PATH \
  --model-path HF_MODEL_PATH \
  --model-name YOUR_MODEL_NAME \
  --output-npz results/gradient_influence/YOUR_MODEL_NAME.npz
```

`--dataset-path` can point to a `.jsonl` file, a `.txt` file, or a directory of `.txt` files. Directory mode first looks for `slice_*.txt`, then falls back to `*.txt`.

By default, the script uses `--max-samples 1000`, `--context-len 8192`, `--target logit_sum`, and `--target-last-n-tokens 20`.

We use `Llama-3.1-8B` in the paper.

## Output

The script only writes a compressed NPZ file. It contains:

- `distance`: relative distance to the last token. `0` is the last token, `1` is one token before it, and so on.
- `mean_grad_norm`: mean gradient norm at each relative distance, averaged over all loaded samples.
- `sum_grad_norm`: summed gradient norm at each relative distance before averaging.
- `sample_count`: number of samples contributing to each distance.
- `model_name`: model name provided by `--model-name`.
- `context_len`: context length used for the curve.
- `target`: target mode. The paper uses `logit_sum`.
- `target_last_n_tokens`: number of final tokens averaged before computing the target.
