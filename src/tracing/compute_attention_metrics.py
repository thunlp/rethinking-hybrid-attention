#!/usr/bin/env python3
"""Compute attention entropy for one checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable.
    tqdm = None


def _load_target_heads(path: Path) -> list[dict[str, int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    heads = payload.get("target_heads", payload)
    parsed: list[dict[str, int]] = []
    for item in heads:
        parsed.append({"layer": int(item["layer"]), "head": int(item["head"])})
    if not parsed:
        raise ValueError("target_heads is empty")
    return parsed


def _head_key(layer: int, head: int) -> str:
    return f"L{layer}_H{head}"


def _init_per_head_stats(target_heads: list[dict[str, int]]) -> dict[str, dict[str, list[float]]]:
    return {
        _head_key(item["layer"], item["head"]): {
            "layer": item["layer"],
            "head": item["head"],
            "entropy": [],
            "entropy_norm": [],
        }
        for item in target_heads
    }


def _summarize_per_head_stats(
    per_head_stats: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for item in per_head_stats.values():
        if not item["entropy"]:
            summarized.append(
                {
                    "layer": item["layer"],
                    "head": item["head"],
                    "attention_entropy": None,
                    "attention_entropy_norm": None,
                }
            )
            continue
        summarized.append(
            {
                "layer": item["layer"],
                "head": item["head"],
                "attention_entropy": float(np.mean(item["entropy"])),
                "attention_entropy_norm": float(np.mean(item["entropy_norm"])),
            }
        )
    summarized.sort(key=lambda row: (row["layer"], row["head"]))
    return summarized


def _load_long_texts(path: Path, max_samples: int) -> list[str]:
    texts: list[str] = []
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
                for key in ("input", "prompt", "text", "context"):
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        texts.append(value)
                        break
                if len(texts) >= max_samples:
                    break
        return texts

    raise ValueError("dataset must be .txt or .jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True, help="HF checkpoint path.")
    parser.add_argument("--target-heads-json", type=str, required=True, help="Target heads JSON.")
    parser.add_argument("--dataset", type=str, required=True, help="Long text dataset (.txt or .jsonl).")
    parser.add_argument("--output-json", type=str, required=True, help="Output metrics JSON path.")
    parser.add_argument("--max-samples", type=int, default=500, help="Max dataset samples.")
    parser.add_argument("--max-input-tokens", type=int, default=8192, help="Max tokens per sample.")
    parser.add_argument("--tail-queries", type=int, default=4, help="Use last N query token positions.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Numerical epsilon for entropy.")
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
    parser.add_argument(
        "--disable-sample-progress",
        action="store_true",
        help="Disable per-sample progress display.",
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


def _clear_trust_remote_code_cache(model_path: str) -> None:
    module_name = Path(model_path).name
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / module_name
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def _seed_trust_remote_code_cache(model_path: str) -> None:
    """Work around dynamic-module copies that miss sibling remote-code files."""
    source_dir = Path(model_path)
    module_name = source_dir.name
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / module_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.py"):
        shutil.copy2(source, cache_dir / source.name)


def _load_model(
    model_path: str,
    device: torch.device,
    use_cuda: bool,
    device_map_mode: str,
    max_memory_gb_per_gpu: int | None,
) -> tuple[Any, Any, torch.device]:
    _clear_trust_remote_code_cache(model_path)
    _seed_trust_remote_code_cache(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    if hasattr(config, "eval_sliding_window"):
        config.eval_sliding_window = None
    if hasattr(config, "eval_attention_sink_size"):
        config.eval_attention_sink_size = 0

    dtype = torch.bfloat16 if use_cuda else torch.float32
    use_device_map = device_map_mode != "none" and use_cuda and torch.cuda.device_count() > 1
    model_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if use_device_map:
        model_kwargs["device_map"] = device_map_mode
        max_memory = _build_max_memory(max_memory_gb_per_gpu)
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        model_kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()
    input_device = _infer_input_device(model)
    if use_device_map:
        print(f"[attention] hf_device_map summary: {_summarize_device_map(model)}", flush=True)
    return tokenizer, model, input_device


def _compute_for_checkpoint(
    hf_path: str,
    target_heads: list[dict[str, int]],
    texts: list[str],
    max_input_tokens: int,
    tail_queries: int,
    eps: float,
    device: torch.device,
    use_cuda: bool,
    device_map_mode: str,
    max_memory_gb_per_gpu: int | None,
    disable_sample_progress: bool,
) -> dict[str, Any]:
    tokenizer, model, input_device = _load_model(
        hf_path,
        device=device,
        use_cuda=use_cuda,
        device_map_mode=device_map_mode,
        max_memory_gb_per_gpu=max_memory_gb_per_gpu,
    )

    entropy_values: list[float] = []
    entropy_norm_values: list[float] = []
    per_head_stats = _init_per_head_stats(target_heads)
    used_samples = 0
    total_samples = len(texts)

    use_tqdm = (
        tqdm is not None
        and not disable_sample_progress
        and total_samples > 0
        and sys.stderr.isatty()
    )
    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            total=total_samples,
            desc="[attention] samples",
            leave=False,
            dynamic_ncols=True,
        )

    with torch.no_grad():
        for sample_idx, text in enumerate(texts, start=1):
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            attentions = outputs.attentions
            if attentions is None or len(attentions) == 0:
                if progress_bar is not None:
                    progress_bar.update(1)
                continue

            seq_len = input_ids.shape[-1]
            start_query = max(0, seq_len - tail_queries)
            query_positions = list(range(start_query, seq_len))

            for head_spec in target_heads:
                layer_idx = head_spec["layer"]
                head_idx = head_spec["head"]
                if layer_idx >= len(attentions):
                    continue

                layer_attn = attentions[layer_idx]
                if layer_attn is None:
                    continue
                if head_idx >= layer_attn.shape[1]:
                    continue

                for query_idx in query_positions:
                    probs = layer_attn[0, head_idx, query_idx, : query_idx + 1].float()
                    probs = probs / (probs.sum() + eps)
                    probs = torch.clamp(probs, min=eps)

                    entropy = -(probs * torch.log(probs)).sum().item()
                    entropy_norm = entropy / float(np.log(len(probs)))

                    entropy_values.append(entropy)
                    entropy_norm_values.append(entropy_norm)

                    head_key = _head_key(layer_idx, head_idx)
                    if head_key in per_head_stats:
                        per_head_stats[head_key]["entropy"].append(entropy)
                        per_head_stats[head_key]["entropy_norm"].append(entropy_norm)

            used_samples += 1
            del outputs, attentions
            if use_cuda:
                torch.cuda.empty_cache()
            if progress_bar is not None:
                progress_bar.update(1)
                if sample_idx % 50 == 0:
                    progress_bar.set_postfix({"used": used_samples}, refresh=False)
            elif sample_idx % 100 == 0:
                print(
                    f"[attention] progress: {sample_idx}/{total_samples} samples (used={used_samples})",
                    flush=True,
                )

    if progress_bar is not None:
        progress_bar.close()

    del model
    if use_cuda:
        torch.cuda.empty_cache()

    if not entropy_values:
        return {
            "samples_used": used_samples,
            "attention_entropy": None,
            "attention_entropy_norm": None,
            "per_head_metrics": _summarize_per_head_stats(per_head_stats),
        }

    return {
        "samples_used": used_samples,
        "attention_entropy": float(np.mean(entropy_values)),
        "attention_entropy_norm": float(np.mean(entropy_norm_values)),
        "per_head_metrics": _summarize_per_head_stats(per_head_stats),
    }


def main() -> None:
    args = parse_args()

    target_heads = _load_target_heads(Path(args.target_heads_json))
    texts = _load_long_texts(Path(args.dataset), max_samples=args.max_samples)
    if not texts:
        raise ValueError("no text samples loaded from dataset")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_device_map = args.device_map != "none" and use_cuda and torch.cuda.device_count() > 1
    print(
        f"[attention] model={args.model_path}, dataset_samples={len(texts)}, device={device}",
        flush=True,
    )
    if use_device_map:
        print(
            f"[attention] using device_map={args.device_map} across {torch.cuda.device_count()} GPUs "
            f"(max_memory_gb_per_gpu={args.max_memory_gb_per_gpu})",
            flush=True,
        )
    elif args.device_map != "none":
        print(
            f"[attention] requested device_map={args.device_map} but it is unavailable "
            "(need CUDA available and >=2 visible GPUs); fallback to single-device mode.",
            flush=True,
        )

    metrics = _compute_for_checkpoint(
        hf_path=args.model_path,
        target_heads=target_heads,
        texts=texts,
        max_input_tokens=args.max_input_tokens,
        tail_queries=args.tail_queries,
        eps=args.eps,
        device=device,
        use_cuda=use_cuda,
        device_map_mode=args.device_map,
        max_memory_gb_per_gpu=args.max_memory_gb_per_gpu,
        disable_sample_progress=args.disable_sample_progress,
    )

    payload = {
        "model_path": args.model_path,
        "target_heads_json": args.target_heads_json,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "max_input_tokens": args.max_input_tokens,
        "tail_queries": args.tail_queries,
        **metrics,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(
        f"[attention] saved metrics -> {output_path}; "
        f"entropy={payload['attention_entropy_norm']}",
        flush=True,
    )
    for head_metrics in metrics.get("per_head_metrics", []):
        print(
            f"[attention]   head L{head_metrics['layer']} H{head_metrics['head']}: "
            f"entropy={head_metrics['attention_entropy_norm']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
