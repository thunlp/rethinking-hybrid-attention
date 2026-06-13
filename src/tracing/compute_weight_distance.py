#!/usr/bin/env python3
"""Compute target-head Wq/Wk distance between two checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:  # pragma: no cover
    load_safetensors_file = None


LAYER_REGEX = re.compile(r"layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True, help="Current HF checkpoint path.")
    parser.add_argument("--reference-model-path", type=str, required=True, help="Reference HF checkpoint path.")
    parser.add_argument("--target-heads-json", type=str, required=True, help="target_heads.json path.")
    parser.add_argument("--output-json", type=str, required=True, help="Output weight-distance JSON path.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Epsilon for relative distance.")
    return parser.parse_args()


def _load_target_heads(path: Path) -> list[dict[str, int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    heads = payload.get("target_heads", payload)
    return [{"layer": int(h["layer"]), "head": int(h["head"])} for h in heads]


def _extract_layer_idx(name: str) -> int | None:
    match = LAYER_REGEX.search(name)
    if not match:
        return None
    return int(match.group(1))


def _collect_qk_weights(model: Any) -> dict[int, dict[str, torch.Tensor]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    if isinstance(model, dict):
        iterator = model.items()
    else:
        iterator = model.named_parameters()
    for name, param in iterator:
        if not name.endswith(".weight"):
            continue
        if ".self_attn." not in name:
            continue
        layer_idx = _extract_layer_idx(name)
        if layer_idx is None:
            continue

        layer_entry = result.setdefault(layer_idx, {})
        if name.endswith("q_proj.weight"):
            layer_entry["q"] = param.detach().float().cpu()
        elif name.endswith("k_proj.weight"):
            layer_entry["k"] = param.detach().float().cpu()
    return result


def _load_model(path: str, device: torch.device) -> Any:
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"HF checkpoint path does not exist: {path}")
    state_dict: dict[str, torch.Tensor] = {}
    safetensor_paths = sorted(local_path.glob("*.safetensors"))
    if safetensor_paths:
        if load_safetensors_file is None:
            raise ImportError("safetensors is required to read safetensors checkpoints")
        for shard_path in safetensor_paths:
            state_dict.update(load_safetensors_file(str(shard_path), device=str(device)))
        return state_dict

    bin_paths = sorted(local_path.glob("pytorch_model*.bin"))
    if not bin_paths:
        bin_paths = sorted(local_path.glob("model*.bin"))
    if not bin_paths:
        raise FileNotFoundError(f"No model weight files found under {path}")
    for shard_path in bin_paths:
        payload = torch.load(shard_path, map_location=device)
        if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        state_dict.update(payload)
    return state_dict


def _slice_q(weight: torch.Tensor, head_idx: int, head_dim: int) -> torch.Tensor:
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim
    return weight[start:end, :]


def _slice_k(
    weight: torch.Tensor,
    head_idx: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    group_size = max(1, num_attention_heads // num_key_value_heads)
    kv_head = min(num_key_value_heads - 1, head_idx // group_size)
    start = kv_head * head_dim
    end = (kv_head + 1) * head_dim
    return weight[start:end, :]


def _distance(curr: torch.Tensor, init: torch.Tensor, eps: float) -> tuple[float, float]:
    diff = curr - init
    l2 = torch.norm(diff, p=2).item()
    rel = l2 / (torch.norm(init, p=2).item() + eps)
    return l2, rel


def main() -> None:
    args = parse_args()
    target_heads = _load_target_heads(Path(args.target_heads_json))
    if not target_heads:
        raise ValueError("target_heads is empty")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[weight] model={args.model_path}, reference={args.reference_model_path}, device={device}", flush=True)

    ref_config = AutoConfig.from_pretrained(args.reference_model_path, trust_remote_code=True, local_files_only=True)
    num_attention_heads = int(getattr(ref_config, "num_attention_heads"))
    num_key_value_heads = int(getattr(ref_config, "num_key_value_heads", num_attention_heads))
    hidden_size = int(getattr(ref_config, "hidden_size"))
    head_dim = hidden_size // num_attention_heads

    ref_model = _load_model(args.reference_model_path, device=device)
    ref_qk = _collect_qk_weights(ref_model)
    del ref_model
    if use_cuda:
        torch.cuda.empty_cache()

    model = _load_model(args.model_path, device=device)
    curr_qk = _collect_qk_weights(model)
    del model
    if use_cuda:
        torch.cuda.empty_cache()

    per_head: list[dict[str, Any]] = []
    for head in target_heads:
        layer_idx = head["layer"]
        head_idx = head["head"]
        if layer_idx not in ref_qk or layer_idx not in curr_qk:
            continue
        if "q" not in ref_qk[layer_idx] or "q" not in curr_qk[layer_idx]:
            continue
        if "k" not in ref_qk[layer_idx] or "k" not in curr_qk[layer_idx]:
            continue

        ref_q = _slice_q(ref_qk[layer_idx]["q"], head_idx=head_idx, head_dim=head_dim)
        curr_q = _slice_q(curr_qk[layer_idx]["q"], head_idx=head_idx, head_dim=head_dim)
        ref_k = _slice_k(
            ref_qk[layer_idx]["k"],
            head_idx=head_idx,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        curr_k = _slice_k(
            curr_qk[layer_idx]["k"],
            head_idx=head_idx,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        q_l2, q_rel = _distance(curr_q, ref_q, eps=args.eps)
        k_l2, k_rel = _distance(curr_k, ref_k, eps=args.eps)
        per_head.append(
            {
                "layer": layer_idx,
                "head": head_idx,
                "q_l2": q_l2,
                "k_l2": k_l2,
                "q_rel_l2": q_rel,
                "k_rel_l2": k_rel,
                "qk_l2": q_l2 + k_l2,
                "qk_rel_l2": q_rel + k_rel,
            }
        )

    if not per_head:
        raise RuntimeError("no usable target heads found in both checkpoints")

    payload = {
        "model_path": args.model_path,
        "reference_model_path": args.reference_model_path,
        "target_heads_json": args.target_heads_json,
        "qk_l2": float(np.mean([x["qk_l2"] for x in per_head])),
        "qk_rel_l2": float(np.mean([x["qk_rel_l2"] for x in per_head])),
        "num_target_heads_used": len(per_head),
        "per_head": per_head,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(
        f"[weight] saved metrics -> {output_path}; "
        f"qk_l2={payload['qk_l2']:.6f}, qk_rel_l2={payload['qk_rel_l2']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
