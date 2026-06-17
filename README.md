# Rethinking the Role of Efficient Attention in Hybrid Architectures

This repository contains the open-source materials for [Rethinking the Role of Efficient Attention in Hybrid Architectures](https://arxiv.org/abs/2606.15378).

## Paper Abstract

Modern language models increasingly adopt hybrid architectures that combine full attention with efficient attention modules, such as sliding-window attention (SWA) and recurrent sequence mixers. However, how these efficient modules shape model capabilities remains poorly understood. To address this gap, we conduct a systematic analysis across hybrid architectures from three perspectives: scaling behavior, mechanism analysis, and architecture design. First, from a scaling perspective, we find that efficient-attention design primarily affects how fast long-context capability emerges, while different hybrids eventually converge to comparable long-context performance under sufficient training. Second, mechanistically, we show that long-range retrieval is mainly carried by full attention, whereas efficient attention shapes its optimization trajectory. This explains a counter-intuitive phenomenon we call *Large-Window Laziness*: larger SWA windows can delay the formation of retrieval heads in full-attention layers. Third, guided by this mechanism, we show that applying NoPE to only the full-attention layers of a small-window SWA hybrid substantially improves long-context performance with negligible impact on short-context performance.

## Repository Overview

- `src/scaling/`: Scaling law fitting.
- `src/probing/`: Layer-wise probing analysis.
- `src/gradient/`: Gradient profiling analysis.
- `src/tracing/`: Retrieval head tracing analysis.

## Installation

Install the core analysis dependencies:

```bash
pip install -r requirements.txt
```

Some released checkpoints rely on architecture-specific inference kernels.
These packages are sensitive to the PyTorch/CUDA/GPU environment, so install
them manually only when needed. See `requirements-optional.txt` for the optional
kernel list and installation notes.

For faster SWA inference, we use the FlashAttention implementation from
[Dao-AILab/flash-attention#1819](https://github.com/Dao-AILab/flash-attention/pull/1819):

```bash
pip install "flash-attn @ git+https://github.com/aoxy/flash-attention.git@feature/attention_with_sink"
```

For Lightning/GDN-style models, install the linear-attention kernels manually:

```bash
pip install "flash-linear-attention>=0.3.1" einops
pip install lightning_attn==0.0.5 --no-deps
```

## Usage

### Scaling Law Fitting

Prepare the input data as a TSV file with the columns `model_size`, `token_multiplier`, `n`, and `loss`. Use `--longppl` for LongPPL fitting and `--validate` to hold out the largest model for validation. We use the `no_constant` formula in the paper.

```bash
cd src/scaling
python fit_scaling.py --input data/swa128_longppl.tsv --formula no_constant --validate --longppl
```

### Receptive Field Constraint Analysis

We provide the model checkpoints used for the receptive field constraint analysis:

[Rethinking Hybrid Constraint Models](https://huggingface.co/EdenQiao/Rethinking-Hybrid-Constraint)

The receptive field constraint is implemented in `modeling.py`. We apply the same constraint method to the SWA and Lightning hybrid models.

### Layer-Wise Probing Analysis

The probing code in `src/probing/` follows a three-stage pipeline:

- `01_prepare_data.py`: Prepare NIAH-like probing datasets.
- `02_extract_hiddenstate.py`: Run the model and extract per-layer hidden states.
- `03_train_probing.py`: Train classifiers on the extracted hidden states and print layer-wise accuracy.

The recommended entry point is `run_probing_pipeline.py`.

```bash
cd src/probing
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

### Gradient Profiling Analysis

The standalone gradient influence script in `src/gradient/` computes token-distance gradient norms and saves the results as a compressed NPZ file.

```bash
cd src/gradient
python run_gradient_influence.py \
  --dataset-path DATASET_PATH \
  --model-path HF_MODEL_PATH \
  --model-name YOUR_MODEL_NAME \
  --output-npz results/gradient_influence/YOUR_MODEL_NAME.npz
```

### Retrieval Head Tracing Analysis

The retrieval head tracing scripts in `src/tracing/` assume the checkpoints have already been converted to Hugging Face format.

Find retrieval heads on NIAH-style validation data:

```bash
cd src/tracing
python find_retrieval_heads.py \
  --model-path HF_MODEL_PATH \
  --niah-jsonl data/niah.jsonl \
  --output-json results/tracing/target_heads.json
```

Compute attention entropy for selected heads:

```bash
python compute_attention_metrics.py \
  --model-path HF_MODEL_PATH \
  --target-heads-json results/tracing/target_heads.json \
  --dataset data/niah.jsonl \
  --output-json results/tracing/attention_metrics.json
```

Compute Q/K weight distance for selected heads between a checkpoint and a reference checkpoint:

```bash
python compute_weight_distance.py \
  --model-path HF_MODEL_PATH \
  --reference-model-path REFERENCE_HF_MODEL_PATH \
  --target-heads-json results/tracing/target_heads.json \
  --output-json results/tracing/weight_distance.json
```

## Model Links

- [Rethinking Hybrid ScalingLaw](https://huggingface.co/EdenQiao/Rethinking-Hybrid-ScalingLaw)
- [Rethinking Hybrid Constraint](https://huggingface.co/EdenQiao/Rethinking-Hybrid-Constraint)

### S5 Models

| Model | 16K | 32K |
| --- | --- | --- |
| Full | [Rethinking-Hybrid-S5-full-16k-5b](https://huggingface.co/EdenQiao/Rethinking-Hybrid-S5-full-100b) | [Rethinking-Hybrid-S5-full-32k-5b](https://huggingface.co/EdenQiao/Rethinking-Hybrid-S5-full-100b-32k-5b) |
| SWA-128 | [Rethinking-Hybrid-S5-SWA-128-16k-5b](https://huggingface.co/EdenQiao/Rethinking-Hybrid-S5-SWA-128-100b) | [Rethinking-Hybrid-S5-SWA-128-32k-5b](https://huggingface.co/EdenQiao/Rethinking-Hybrid-S5-SWA-128-100b-32k-5b) |
| SWA-128-NoPE | [Rethinking-Hybrid-S5-SWA-128-Nope-16k-5b](https://huggingface.co/EdenQiao/Rethinking-Hybrid-S5-SWA-128-Nope-100b) | [Rethinking-Hybrid-S5-SWA-128-Nope-32k-5b](https://huggingface.co/EdenQiao/Rethinking-Hybrid-S5-SWA-128-Nope-100b-32k-5b) |
