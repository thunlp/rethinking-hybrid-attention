# Layer-wise Probing Analysis

This directory contains the layer-wise probing pipeline.

## Pipeline

The recommended entry point is `run_probing_pipeline.py`, which runs the full pipeline:

1. `01_prepare_data.py` builds the probing dataset.
2. `02_extract_hiddenstate.py` extracts per-layer hidden-state from the model.
3. `03_train_probing.py` trains a classifier for each layer and prints the layer-wise probing results.

Shared dataset naming and output path conventions are defined in `common.py`.

## Usage

```bash
python run_probing_pipeline.py \
  --model_path HF_MODEL_PATH \
  --model_alias YOUR_MODEL_ALIAS \
  --haystack_type repeat \
  --num_samples 10000 \
  --seq_len 16000 \
  --num_classes 8 \
  --difficulty hard \
  --feature_type residual \
  --classifier logistic
```

Use `--skip_data_prep`, `--skip_extraction`, or `--skip_training` to resume from an intermediate step.

## Outputs

By default, outputs are written under the repository root:

- `data/probing/`: generated probing datasets.
- `results/probing/`: extracted hidden-state features.

The output paths are derived from the same dataset specification.

## Individual Steps

Each step can also be run independently when debugging or reusing intermediate files. The pipeline passes the same dataset metadata to the data-preparation and activation-extraction steps so that generated datasets and activations follow the same directory layout.
