import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from argparse import ArgumentParser
from common import (
    build_hiddenstate_file,
    DIFFICULTY_CHOICES,
    extract_dataset_name,
    get_default_results_dir,
    parse_dataset_name,
    resolve_dataset_spec,
)

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, default="./data/probing_data.json")
parser.add_argument("--output_dir", type=str, default=str(get_default_results_dir(__file__)))
parser.add_argument("--model_alias", type=str, required=True, help="e.g., swa_128, base")
parser.add_argument("--max_len", type=int, default=20000)
parser.add_argument("--method_name", type=str, default=None, help="method name (e.g., niah_single_1)")
parser.add_argument("--seq_len", type=int, default=None, help="sequence length")
parser.add_argument("--num_samples", type=int, default=None, help="number of samples")
parser.add_argument("--num_classes", type=int, default=None, help="number of classes used to build the dataset")
parser.add_argument(
    "--difficulty",
    type=str,
    default=None,
    choices=DIFFICULTY_CHOICES,
    help="difficulty tier for dataset path layout",
)
parser.add_argument(
    "--feature_type",
    type=str,
    default="residual",
    choices=["residual", "delta"],
    help="Feature type: residual keeps the post-layer hidden state, delta keeps only the current layer contribution h_l - h_{l-1}",
)
args = parser.parse_args()

def main():
    # Extract dataset name from data path
    dataset_name = extract_dataset_name(args.data_path)
    print(f"Dataset name: {dataset_name or 'unknown'}")
    
    # 1. Load Data
    with open(args.data_path, "r") as f:
        dataset = json.load(f)
    
    # 2. Load Model
    print(f"Loading model: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if hasattr(config, "eval_sliding_window"):
        config.eval_sliding_window = None
    if hasattr(config, "eval_attention_sink_size"):
        config.eval_attention_sink_size = 0
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        config=config, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        local_files_only=True,
        device_map="cuda",
    ).eval()
    
    num_transformer_layers = config.num_hidden_layers
    layer_activations = {i: [] for i in range(num_transformer_layers)}
    labels = []
    
    print(f"Extracting features for {len(dataset)} samples...")
    
    for sample in tqdm(dataset):
        text = sample["text"]
        label = int(sample["label"])  # Ensure label is integer (0-3 for 4-class classification)
        
        encoded = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=args.max_len, 
            truncation=True,
            padding=False
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        
        # Forward Pass
        with torch.no_grad():
            outputs = model(
                input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True, 
                use_cache=False
            )
        
        hidden_states = outputs.hidden_states
        for transformer_layer_idx in range(num_transformer_layers):
            prev_hidden_state = hidden_states[transformer_layer_idx]
            current_hidden_state = hidden_states[transformer_layer_idx + 1]
            prev_last_token_vec = prev_hidden_state[0, -1, :].cpu().float().numpy()
            current_last_token_vec = current_hidden_state[0, -1, :].cpu().float().numpy()

            if args.feature_type == "residual":
                feature_vec = current_last_token_vec
            else:
                feature_vec = current_last_token_vec - prev_last_token_vec

            layer_activations[transformer_layer_idx].append(feature_vec)
        
        labels.append(label)
        

    # 3. Determine output path structure
    parsed_spec = parse_dataset_name(dataset_name) if dataset_name else None
    num_classes = args.num_classes
    if num_classes is None and parsed_spec is not None and parsed_spec.num_classes is not None:
        num_classes = parsed_spec.num_classes
    if num_classes is None and dataset:
        num_classes = dataset[0].get("num_classes")
    spec = resolve_dataset_spec(
        parsed_spec=parsed_spec,
        method_name=args.method_name,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_classes=num_classes,
        difficulty=args.difficulty,
    )
    save_path = build_hiddenstate_file(
        output_dir=Path(args.output_dir),
        spec=spec,
        model_alias=args.model_alias,
        feature_type=args.feature_type,
    )
    if spec is None and dataset_name:
        suffix = "" if args.feature_type == "residual" else f"_{args.feature_type}"
        save_path = Path(args.output_dir) / f"acts_{args.model_alias}{suffix}_{dataset_name}.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "labels": np.array(labels),
        "feature_type": np.array(args.feature_type),
    }
    if num_classes is not None:
        save_dict["num_classes"] = np.array(num_classes)
    for k, v in layer_activations.items():
        feature_key = f"layer_{k}" if args.feature_type == "residual" else f"delta_layer_{k}"
        save_dict[feature_key] = np.stack(v) # shape [num_samples, hidden_dim]
        
    np.savez_compressed(str(save_path), **save_dict)
    print(f"Saved activations to {save_path}")

if __name__ == "__main__":
    main()