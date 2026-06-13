# Retrieval Head Tracing Analysis

This directory contains the core scripts for retrieval-head tracing analysis.

## Workflow

1. `find_retrieval_heads.py` identifies retrieval heads on NIAH-style validation data.
2. `compute_attention_metrics.py` computes attention metrics for the selected heads on one checkpoint.
3. `compute_weight_distance.py` computes Q/K weight distance for the selected heads between one checkpoint and a reference checkpoint.

In the paper, we first utilize `find_retrieval_heads.py` to identify the retrieval heads of the final checkpoint, and then use `compute_attention_metrics.py` and `compute_weight_distance.py` to compute the attention metrics and weight distance for the intermediate checkpoints in the training process.

## Find Retrieval Heads

```bash
python find_retrieval_heads.py \
  --model-path HF_MODEL_PATH \
  --niah-jsonl data/niah.jsonl \
  --output-json results/tracing/target_heads.json
```

This script finds answer-token positions in NIAH samples, measures how much each layer/head attends to those positions from the query token, and saves the top retrieval heads.

The output JSON contains `target_heads` and the full `layer_head_scores` matrix.

## Compute Attention Metrics

```bash
python compute_attention_metrics.py \
  --model-path HF_MODEL_PATH \
  --target-heads-json results/tracing/target_heads.json \
  --dataset data/niah.jsonl \
  --output-json results/tracing/attention_metrics.json
```

This script computes attention entropy and normalized attention entropy for the selected heads.

The output JSON contains aggregate metrics and `per_head_metrics`.

## Compute Weight Distance

```bash
python compute_weight_distance.py \
  --model-path HF_MODEL_PATH \
  --reference-model-path REFERENCE_HF_MODEL_PATH \
  --target-heads-json results/tracing/target_heads.json \
  --output-json results/tracing/weight_distance.json
```

This script extracts the Q/K projection weights for the selected heads and compares the current checkpoint to a reference checkpoint.

The output JSON contains average `qk_l2`, average `qk_rel_l2`, and per-head Q/K distance metrics.
