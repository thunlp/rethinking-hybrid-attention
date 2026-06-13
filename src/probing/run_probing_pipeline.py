#!/usr/bin/env python3
"""
Pipeline script to run the complete probing workflow:
1. Prepare data (01_prepare_data.py)
2. Extract hidden states (02_extract_hiddenstate.py)
3. Train probing classifier (03_train_probing.py)
"""

import sys
import subprocess
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROBING_DIR = SCRIPT_DIR
sys.path.insert(0, str(PROBING_DIR))
from common import (
    build_dataset_name,
    build_dataset_spec,
    build_hiddenstate_file,
    DIFFICULTY_CHOICES,
    get_default_data_dir,
    get_default_results_dir,
)

parser = argparse.ArgumentParser(description="Run complete probing pipeline")
# Step 1: Data preparation arguments
parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples to generate")
parser.add_argument("--seq_len", type=int, default=16000, help="Target sequence length")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--data_dir",
    type=str,
    default=None,
    help="Directory to save/load probing data (default: <repo>/data/probing)",
)
parser.add_argument("--haystack_type", type=str, default="repeat", choices=["repeat", "essay"],
                    help="Type of haystack: repeat/essay")
parser.add_argument("--essay_path", type=str, default=None,
                    help="Path to PaulGrahamEssays.json (default: auto-detect)")
parser.add_argument("--num_classes", type=int, default=4, help="Number of candidate classes")
parser.add_argument(
    "--difficulty",
    type=str,
    default="easy",
    choices=DIFFICULTY_CHOICES,
    help="Difficulty tier for magic-number classification.",
)
parser.add_argument("--depth_min", type=int, default=0, help="Minimum insertion depth percentage")
parser.add_argument("--depth_max", type=int, default=99, help="Maximum insertion depth percentage")
parser.add_argument("--depth_step", type=int, default=3, help="Insertion depth step size")

# Step 2: Model extraction arguments
parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
parser.add_argument("--model_alias", type=str, required=True, help="Model alias (e.g., s3.128.2, s3.sink)")
parser.add_argument("--max_len", type=int, default=20000, help="Max sequence length for tokenization")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save hidden states (default: <repo>/results/probing)",
)
parser.add_argument(
    "--feature_type",
    type=str,
    default="residual",
    choices=["residual", "delta"],
    help="Feature type for probing: residual uses post-layer hidden states, delta uses h_l - h_{l-1}",
)
parser.add_argument(
    "--classifier",
    type=str,
    default="logistic",
    choices=["logistic", "mlp", "svm", "random_forest", "gradient_boosting", "naive_bayes", "knn"],
    help="Classifier type used in probing training.",
)

# Step 3: Training arguments (mostly use defaults)
parser.add_argument("--skip_data_prep", action="store_true", help="Skip data preparation if data already exists")
parser.add_argument("--skip_extraction", action="store_true", help="Skip extraction if hidden states already exist")
parser.add_argument("--skip_training", action="store_true", help="Skip training if results already exist")

args = parser.parse_args()

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=PROBING_DIR)
    
    if result.returncode != 0:
        print(f"\n❌ Error in {description}")
        sys.exit(1)
    
    print(f"\n✅ Successfully completed: {description}\n")
    return result

def main():
    if args.data_dir is None:
        data_dir = get_default_data_dir(__file__)
    else:
        data_dir = Path(args.data_dir)
    
    if args.output_dir is None:
        output_dir = get_default_results_dir(__file__)
    else:
        output_dir = Path(args.output_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    spec = build_dataset_spec(
        haystack_type=args.haystack_type,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        difficulty=args.difficulty,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_step=args.depth_step,
    )
    method_name = spec.method_name
    dataset_name = build_dataset_name(spec)
    data_path = data_dir / f"probing_data_{dataset_name}.json"
    acts_file = build_hiddenstate_file(
        output_dir=output_dir,
        spec=spec,
        model_alias=args.model_alias,
        feature_type=args.feature_type,
    )
    
    # Step 1: Prepare data
    if not args.skip_data_prep and not data_path.exists():
        print(f"\n📝 Step 1: Preparing data...")
        cmd = [
            sys.executable,
            str(PROBING_DIR / "01_prepare_data.py"),
            "--save_dir", str(data_dir),
            "--num_samples", str(args.num_samples),
            "--seq_len", str(args.seq_len),
            "--seed", str(args.seed),
            "--haystack_type", args.haystack_type,
            "--num_classes", str(args.num_classes),
            "--difficulty", args.difficulty,
            "--depth_min", str(args.depth_min),
            "--depth_max", str(args.depth_max),
            "--depth_step", str(args.depth_step),
        ]
        if args.essay_path:
            cmd.extend(["--essay_path", args.essay_path])
        run_command(cmd, "Data Preparation")
    elif data_path.exists():
        print(f"\n⏭️  Skipping data preparation (file exists: {data_path})")
    else:
        print(f"\n⏭️  Skipping data preparation (--skip_data_prep flag)")
    
    # Step 2: Extract hidden states
    if not args.skip_extraction:
        if acts_file.exists():
            print(f"\n♻️  Re-running extraction and overwriting: {acts_file}")
        print(f"\n🔍 Step 2: Extracting hidden states...")
        cmd = [
            sys.executable,
            str(PROBING_DIR / "02_extract_hiddenstate.py"),
            "--model_path", args.model_path,
            "--data_path", str(data_path),
            "--output_dir", str(output_dir),
            "--model_alias", args.model_alias,
            "--max_len", str(args.max_len),
            "--feature_type", args.feature_type,
            "--method_name", method_name,
            "--seq_len", str(args.seq_len),
            "--num_samples", str(args.num_samples),
            "--num_classes", str(args.num_classes),
            "--difficulty", args.difficulty,
        ]
        run_command(cmd, "Hidden State Extraction")
    else:
        print(f"\n⏭️  Skipping extraction (--skip_extraction flag)")
    
    # Step 3: Train probing classifier
    if not args.skip_training:
        print(f"\n🎯 Step 3: Training probing classifier...")
        cmd = [
            sys.executable,
            str(PROBING_DIR / "03_train_probing.py"),
            "--acts_file", str(acts_file),
            "--model_alias", args.model_alias,
            "--feature_type", args.feature_type,
            "--classifier", args.classifier,
        ]
        run_command(cmd, "Probing Classifier Training")
    else:
        print(f"\n⏭️  Skipping training (--skip_training flag)")
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ Pipeline completed successfully!")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  - Data: {data_path}")
    print(f"  - Hidden states: {acts_file}")
    print(f"\nConfiguration:")
    print(f"  - Method: {method_name} ({args.haystack_type} haystack)")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Samples: {args.num_samples}")
    print(f"  - Sequence length: {args.seq_len}")
    print(f"  - Difficulty: {args.difficulty}")
    print(f"  - Feature type: {args.feature_type}")
    print(f"  - Classifier: {args.classifier}")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()

