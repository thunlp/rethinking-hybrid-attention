#!/usr/bin/env python3
"""Compute token-distance gradient influence curves."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _get_repo_root(reference_file: str) -> Path:
    ref = Path(reference_file).resolve()
    for parent in [ref.parent, *ref.parents]:
        if (parent / "src").exists():
            return parent
    return ref.parents[2]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_dtype(name: str) -> torch.dtype:
    normalized = name.lower().strip()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[normalized]


def _load_texts(path: Path, max_samples: int, text_keys: list[str], seed: int) -> list[str]:
    texts: list[str] = []
    if path.is_dir():
        file_paths = sorted(path.glob("slice_*.txt"))
        if not file_paths:
            file_paths = sorted(path.glob("*.txt"))
        if not file_paths:
            raise ValueError(f"dataset directory has no text files: {path}")

        # Deterministic subset selection for reproducibility.
        rng = random.Random(seed)
        file_paths = list(file_paths)
        rng.shuffle(file_paths)
        selected = file_paths[: max_samples]
        for file_path in selected:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                texts.append(text)
        return texts

    suffix = path.suffix.lower()

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                texts.append(line)
                if len(texts) >= max_samples:
                    break
        return texts

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for key in text_keys:
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        texts.append(value)
                        break
                if len(texts) >= max_samples:
                    break
        return texts

    raise ValueError("--dataset-path must be a directory, .jsonl, or .txt file")


def _infer_input_device(model: Any, fallback_device: torch.device) -> torch.device:
    if hasattr(model, "hf_device_map"):
        device_map = getattr(model, "hf_device_map")
        if isinstance(device_map, dict):
            for _, value in device_map.items():
                if isinstance(value, str) and value.startswith("cuda"):
                    return torch.device(value)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def _clear_trust_remote_code_cache(model_path: str) -> None:
    module_name = Path(model_path).name
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / module_name
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    # Also clear already-imported modules in current process.
    prefixes = (f"transformers_modules.{module_name}", f"transformers_modules.{module_name}.")
    for name in list(sys.modules.keys()):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


def _compute_curve_for_model(
    model_name: str,
    model_path: str,
    texts: list[str],
    context_len: int,
    device: torch.device,
    dtype: torch.dtype,
    target_mode: str,
    target_last_n_tokens: int,
    attn_implementation: str,
    gradient_checkpointing: bool,
    trust_remote_code: bool,
    local_files_only: bool,
    device_map: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if trust_remote_code:
        _clear_trust_remote_code_cache(model_path)
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    # Keep implementation configurable: eager/sdpa/flash_attention_2
    config._attn_implementation = attn_implementation
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = attn_implementation

    model_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
        "torch_dtype": dtype,
    }
    if device_map != "none":
        model_kwargs["device_map"] = device_map
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if device_map == "none":
        model = model.to(device)
    model.eval()
    # We only need d(target)/d(inputs_embeds): disable parameter grads to save memory.
    model.requires_grad_(False)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    input_device = _infer_input_device(model, fallback_device=device)

    sums = np.zeros(context_len, dtype=np.float64)
    counts = np.zeros(context_len, dtype=np.int64)

    emb_layer = model.get_input_embeddings()
    if emb_layer is None:
        raise ValueError(f"model has no embedding layer: {model_name}")

    with torch.enable_grad():
        progress = tqdm(texts, desc=f"[{model_name}] gradients", leave=False)
        for text in progress:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=context_len,
            )
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_device)

            if input_ids.shape[-1] < 2:
                continue

            model.zero_grad(set_to_none=True)
            inputs_embeds = emb_layer(input_ids).detach()
            inputs_embeds.requires_grad_(True)

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            effective_last_n = min(target_last_n_tokens, int(outputs.logits.shape[1]))
            logits = outputs.logits[:, -effective_last_n:, :].mean(dim=1)
            if target_mode == "argmax_logit":
                argmax_idx = logits.argmax(dim=-1, keepdim=True)
                target = logits.gather(dim=-1, index=argmax_idx).sum()
            elif target_mode == "logit_sum":
                target = logits.sum()
            else:
                raise ValueError(f"unsupported compute.target: {target_mode}")

            target.backward()
            grads = inputs_embeds.grad
            if grads is None:
                continue
            norms = torch.linalg.vector_norm(grads[0].float(), ord=2, dim=-1).detach().cpu().numpy()

            seq_len = int(norms.shape[0])
            for dist in range(seq_len):
                token_idx = seq_len - 1 - dist
                sums[dist] += float(norms[token_idx])
                counts[dist] += 1

            del outputs, logits, target, inputs_embeds, grads
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mean_curve = np.full(context_len, np.nan, dtype=np.float64)
    valid = counts > 0
    mean_curve[valid] = sums[valid] / counts[valid]
    return mean_curve, sums, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = _get_repo_root(__file__) / "results" / "gradient_influence" / "gradient_influence.npz"
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to a .jsonl file, .txt file, or directory of .txt files.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Hugging Face model.")
    parser.add_argument("--model-name", type=str, default="model", help="Name stored in the NPZ metadata.")
    parser.add_argument("--output-npz", type=str, default=str(default_output), help="Path to save the compressed NPZ output.")
    parser.add_argument("--text-keys", type=str, default="input,prompt,text,context", help="Comma-separated keys to read from JSONL rows.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum number of samples to load.")
    parser.add_argument("--context-len", type=int, default=8192, help="Fixed context length for the gradient curve.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and computation.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32, float16, or bfloat16.")
    parser.add_argument("--target", type=str, default="logit_sum", choices=["argmax_logit", "logit_sum"])
    parser.add_argument("--target-last-n-tokens", type=int, default=20, help="Average logits over the last N tokens before computing the target.")
    parser.add_argument("--attn-implementation", type=str, default="eager", help="eager, sdpa, or flash_attention_2.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads instead of requiring local files.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", type=str, default="none", help="none, auto, balanced, balanced_low_0, or sequential.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset path not found: {dataset_path}")
    _set_seed(args.seed)
    text_keys = [key.strip() for key in args.text_keys.split(",") if key.strip()]
    texts = _load_texts(
        dataset_path,
        max_samples=args.max_samples,
        text_keys=text_keys,
        seed=args.seed,
    )
    if not texts:
        raise ValueError("no text samples loaded from dataset")
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[grad] warning: CUDA requested but unavailable, fallback to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    dtype = _to_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("[grad] warning: forcing dtype=float32 on CPU.", flush=True)
        dtype = torch.float32
    if args.target_last_n_tokens < 1:
        raise ValueError("--target-last-n-tokens must be >= 1")
    local_files_only = not args.allow_download

    print(
        f"[grad] model={args.model_name}, samples={len(texts)}, context_len={args.context_len}, "
        f"device={device}, dtype={dtype}, target={args.target}, "
        f"target_last_n_tokens={args.target_last_n_tokens}, attn_impl={args.attn_implementation}, "
        f"gradient_checkpointing={args.gradient_checkpointing}, trust_remote_code={args.trust_remote_code}",
        flush=True,
    )

    mean_curve, sum_grad_norm, sample_count = _compute_curve_for_model(
        model_name=args.model_name,
        model_path=args.model_path,
        texts=texts,
        context_len=args.context_len,
        device=device,
        dtype=dtype,
        target_mode=args.target,
        target_last_n_tokens=args.target_last_n_tokens,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
        device_map=args.device_map,
    )

    distance = np.arange(args.context_len, dtype=np.int64)
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_npz),
        distance=distance,
        mean_grad_norm=mean_curve,
        sum_grad_norm=sum_grad_norm,
        sample_count=sample_count,
        model_name=np.array(args.model_name),
        context_len=np.array(args.context_len),
        target=np.array(args.target),
        target_last_n_tokens=np.array(args.target_last_n_tokens),
    )
    print(f"[grad] saved npz -> {output_npz}", flush=True)


if __name__ == "__main__":
    main()
