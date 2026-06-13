#!/usr/bin/env python3
"""Identify retrieval target heads on NIAH-style validation data."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class Sample:
    prompt: str
    needles: list[str]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_prompt(row: dict[str, Any]) -> str:
    for key in ("input", "prompt", "text", "context", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("cannot find prompt field in NIAH row")


def _extract_needles(row: dict[str, Any]) -> list[str]:
    needles: list[str] = []

    outputs = row.get("outputs")
    if isinstance(outputs, list):
        needles.extend([str(x) for x in outputs if str(x).strip()])
    elif isinstance(outputs, str) and outputs.strip():
        needles.append(outputs.strip())

    for key in ("answer", "needle", "target"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            needles.append(value.strip())
        elif isinstance(value, list):
            needles.extend([str(x) for x in value if str(x).strip()])

    # Preserve order while de-duplicating.
    deduped: list[str] = []
    seen: set[str] = set()
    for needle in needles:
        if needle in seen:
            continue
        seen.add(needle)
        deduped.append(needle)
    return deduped


def load_samples(path: Path, max_samples: int) -> list[Sample]:
    rows = _load_jsonl(path)
    samples: list[Sample] = []
    for row in rows:
        prompt = _extract_prompt(row)
        needles = _extract_needles(row)
        samples.append(Sample(prompt=prompt, needles=needles))
        if len(samples) >= max_samples:
            break
    return samples


def _find_subsequence(sequence: list[int], pattern: list[int]) -> list[int]:
    if not sequence or not pattern or len(pattern) > len(sequence):
        return []
    span_positions: list[int] = []
    plen = len(pattern)
    for start in range(0, len(sequence) - plen + 1):
        if sequence[start : start + plen] == pattern:
            span_positions.extend(range(start, start + plen))
    return span_positions


def _needle_positions(tokenizer: AutoTokenizer, input_ids: torch.Tensor, needles: list[str]) -> list[int]:
    token_list = input_ids.tolist()
    positions: set[int] = set()
    for needle in needles:
        variants = [
            needle,
            f" {needle}",
            f"{needle}.",
            f" {needle}.",
            f": {needle}",
            f": {needle}.",
        ]
        for variant in variants:
            needle_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not needle_ids:
                continue
            for pos in _find_subsequence(token_list, needle_ids):
                positions.add(pos)
    return sorted(positions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True, help="HF checkpoint path.")
    parser.add_argument("--niah-jsonl", type=str, required=True, help="NIAH validation JSONL path.")
    parser.add_argument("--output-json", type=str, required=True, help="Output target-head JSON path.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of target heads to select.")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum number of NIAH samples.")
    parser.add_argument("--max-input-tokens", type=int, default=8192, help="Max tokens per sample.")
    parser.add_argument(
        "--query-index-from-end",
        type=int,
        default=2,
        help="Use query token at position len(input)-query_index_from_end.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--device-map",
        type=str,
        default="balanced_low_0",
        choices=["none", "auto", "balanced", "balanced_low_0", "sequential"],
        help="Use HuggingFace device map strategy for multi-GPU sharding.",
    )
    parser.add_argument(
        "--max-memory-gb-per-gpu",
        type=int,
        default=72,
        help="Optional per-GPU memory cap in GiB when device-map=auto.",
    )
    return parser.parse_args()


def _build_max_memory(max_memory_gb_per_gpu: int | None) -> dict[Any, str] | None:
    if max_memory_gb_per_gpu is None:
        return None
    if not torch.cuda.is_available():
        return None
    memory: dict[Any, str] = {idx: f"{max_memory_gb_per_gpu}GiB" for idx in range(torch.cuda.device_count())}
    memory["cpu"] = "64GiB"
    return memory


def _infer_input_device(model: Any) -> torch.device:
    if hasattr(model, "hf_device_map"):
        device_map = getattr(model, "hf_device_map")
        if isinstance(device_map, dict):
            for _, value in device_map.items():
                if isinstance(value, str) and value.startswith("cuda"):
                    return torch.device(value)
    return next(model.parameters()).device


def _summarize_device_map(model: Any) -> str:
    if not hasattr(model, "hf_device_map"):
        return "no hf_device_map"
    device_map = getattr(model, "hf_device_map")
    if not isinstance(device_map, dict):
        return "invalid hf_device_map"
    counts: dict[str, int] = {}
    for _, value in device_map.items():
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    return ", ".join(parts) if parts else "empty hf_device_map"


def _clear_trust_remote_code_cache(model_path: Path) -> None:
    module_name = model_path.name
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / module_name
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def _seed_trust_remote_code_cache(model_path: Path) -> None:
    """Work around dynamic-module copies that miss sibling remote-code files."""
    module_name = model_path.name
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / module_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in model_path.glob("*.py"):
        shutil.copy2(source, cache_dir / source.name)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    dataset_path = Path(args.niah_jsonl)
    output_path = Path(args.output_json)

    samples = load_samples(dataset_path, max_samples=args.max_samples)
    if not samples:
        raise ValueError("no samples loaded from niah dataset")
    print(f"[niah] loaded samples={len(samples)} from {dataset_path}", flush=True)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    use_device_map = args.device_map != "none" and use_cuda and torch.cuda.device_count() > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[niah] loading model from {model_path} on device={device}", flush=True)
    if args.device_map != "none" and not use_device_map:
        print(
            f"[niah] requested device_map={args.device_map} but it is unavailable "
            "(need CUDA available and >=2 visible GPUs); fallback to single-device mode.",
            flush=True,
        )

    _clear_trust_remote_code_cache(model_path)
    _seed_trust_remote_code_cache(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    if hasattr(config, "eval_sliding_window"):
        config.eval_sliding_window = None
    if hasattr(config, "eval_attention_sink_size"):
        config.eval_attention_sink_size = 0

    dtype = torch.bfloat16 if use_cuda else torch.float32
    model_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if use_device_map:
        model_kwargs["device_map"] = args.device_map
        max_memory = _build_max_memory(args.max_memory_gb_per_gpu)
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        model_kwargs["low_cpu_mem_usage"] = True
        print(
            f"[niah] using device_map={args.device_map} across {torch.cuda.device_count()} GPUs "
            f"(max_memory={max_memory})",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs).to(device)
    model.eval()
    if use_device_map:
        print(f"[niah] hf_device_map summary: {_summarize_device_map(model)}", flush=True)
    input_device = _infer_input_device(model)
    print(f"[niah] input tensors will be moved to {input_device}", flush=True)

    layer_head_scores: np.ndarray | None = None
    used_samples = 0
    missing_needle_samples = 0

    with torch.no_grad():
        for idx, sample in enumerate(samples, start=1):
            encoded = tokenizer(
                sample.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_device)

            needle_pos = _needle_positions(tokenizer, input_ids[0].cpu(), sample.needles)
            if not needle_pos:
                missing_needle_samples += 1
                if idx % 5 == 0 or idx == len(samples):
                    print(
                        f"[niah][{idx}/{len(samples)}] skip sample (needle not found), "
                        f"used={used_samples}, missing={missing_needle_samples}",
                        flush=True,
                    )
                continue

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            attentions = outputs.attentions
            if attentions is None or len(attentions) == 0:
                continue
            valid_attentions = [layer for layer in attentions if layer is not None]
            if not valid_attentions:
                continue

            query_index = max(0, input_ids.shape[-1] - args.query_index_from_end)
            query_index = min(query_index, input_ids.shape[-1] - 1)

            if layer_head_scores is None:
                n_layers = len(attentions)
                n_heads = valid_attentions[0].shape[1]
                layer_head_scores = np.zeros((n_layers, n_heads), dtype=np.float64)

            has_valid_layer = False
            for layer_idx, layer_attn in enumerate(attentions):
                if layer_attn is None:
                    continue
                has_valid_layer = True
                # shape: [batch, heads, query, key]
                score_vec = layer_attn[0, :, query_index, :]
                score = score_vec[:, needle_pos].sum(dim=-1).float().cpu().numpy()
                layer_head_scores[layer_idx] += score

            if not has_valid_layer:
                continue
            used_samples += 1
            del outputs, attentions
            if use_cuda:
                torch.cuda.empty_cache()
            if idx % 5 == 0 or idx == len(samples):
                print(
                    f"[niah][{idx}/{len(samples)}] processed, used={used_samples}, missing={missing_needle_samples}",
                    flush=True,
                )

    if layer_head_scores is None or used_samples == 0:
        raise RuntimeError("failed to collect head scores from provided samples")

    layer_head_scores /= used_samples

    flat = []
    for layer_idx in range(layer_head_scores.shape[0]):
        for head_idx in range(layer_head_scores.shape[1]):
            flat.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "score": float(layer_head_scores[layer_idx, head_idx]),
                }
            )
    flat.sort(key=lambda item: item["score"], reverse=True)
    target_heads = flat[: args.top_k]

    payload = {
        "model_path": str(model_path),
        "niah_jsonl": str(dataset_path),
        "top_k": args.top_k,
        "samples_used": used_samples,
        "samples_missing_needle": missing_needle_samples,
        "query_index_from_end": args.query_index_from_end,
        "target_heads": target_heads,
        "layer_head_scores": layer_head_scores.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(
        f"[niah] saved target heads -> {output_path} "
        f"(top_k={args.top_k}, used_samples={used_samples})",
        flush=True,
    )


if __name__ == "__main__":
    main()
