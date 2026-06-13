from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    method_name: str
    num_samples: int
    seq_len: int
    num_classes: int | None = None
    depth_tag: str | None = None
    difficulty: str | None = None


DIFFICULTY_CHOICES = ("easy", "hard")


DATASET_NAME_PATTERN = re.compile(
    r"^(?P<method_name>.+)_(?P<num_samples>\d+)_(?P<seq_len>\d+)"
    r"(?:_(?P<num_classes>\d+)c)?(?:_(?P<depth_tag>d\d+-\d+-s\d+))?(?:_(?P<difficulty>easy|hard))?$"
)


def get_repo_root(reference_file: str) -> Path:
    ref = Path(reference_file).resolve()
    for parent in [ref.parent, *ref.parents]:
        if (parent / "src").exists() and (parent / "data").exists():
            return parent
    # Fallback for typical layout: <root>/src/probing/*.py
    return ref.parents[2]


def get_default_data_dir(reference_file: str) -> Path:
    return get_repo_root(reference_file) / "data" / "probing"


def get_default_results_dir(reference_file: str) -> Path:
    return get_repo_root(reference_file) / "results" / "probing"


def method_name_from_haystack(haystack_type: str) -> str:
    mapping = {
        "repeat": "niah_single_1",
        "essay": "niah_single_2",
    }
    if haystack_type not in mapping:
        raise ValueError(f"unsupported haystack_type: {haystack_type}")
    return mapping[haystack_type]


def build_depths(depth_min: int, depth_max: int, depth_step: int) -> list[int]:
    depths = list(range(depth_min, depth_max + 1, depth_step))
    if depths and depths[-1] != depth_max:
        depths.append(depth_max)
    return sorted(set(depths))


def build_depth_tag(depth_min: int, depth_max: int, depth_step: int) -> str | None:
    if depth_min == 0 and depth_max == 99 and depth_step == 3:
        return None
    return f"d{depth_min}-{depth_max}-s{depth_step}"


def build_dataset_name(spec: DatasetSpec) -> str:
    name = f"{spec.method_name}_{spec.num_samples}_{spec.seq_len}"
    if spec.num_classes is not None:
        name = f"{name}_{spec.num_classes}c"
    if spec.depth_tag:
        name = f"{name}_{spec.depth_tag}"
    if spec.difficulty:
        name = f"{name}_{spec.difficulty}"
    return name


def build_dataset_spec(
    haystack_type: str,
    num_samples: int,
    seq_len: int,
    num_classes: int | None = None,
    difficulty: str | None = None,
    depth_min: int = 0,
    depth_max: int = 99,
    depth_step: int = 3,
) -> DatasetSpec:
    method_name = method_name_from_haystack(haystack_type)
    depth_tag = (
        build_depth_tag(depth_min, depth_max, depth_step)
        if haystack_type == "essay"
        else None
    )
    return DatasetSpec(
        method_name=method_name,
        num_samples=num_samples,
        seq_len=seq_len,
        num_classes=num_classes,
        depth_tag=depth_tag,
        difficulty=difficulty,
    )


def resolve_dataset_spec(
    parsed_spec: DatasetSpec | None,
    method_name: str | None = None,
    num_samples: int | None = None,
    seq_len: int | None = None,
    num_classes: int | None = None,
    depth_tag: str | None = None,
    difficulty: str | None = None,
) -> DatasetSpec | None:
    method_name = method_name or (parsed_spec.method_name if parsed_spec else None)
    num_samples = num_samples or (parsed_spec.num_samples if parsed_spec else None)
    seq_len = seq_len or (parsed_spec.seq_len if parsed_spec else None)
    num_classes = num_classes if num_classes is not None else (parsed_spec.num_classes if parsed_spec else None)
    depth_tag = depth_tag or (parsed_spec.depth_tag if parsed_spec else None)
    difficulty = difficulty or (parsed_spec.difficulty if parsed_spec else None)

    if not (method_name and num_samples and seq_len):
        return None

    return DatasetSpec(
        method_name=method_name,
        num_samples=int(num_samples),
        seq_len=int(seq_len),
        num_classes=int(num_classes) if num_classes is not None else None,
        depth_tag=depth_tag,
        difficulty=difficulty,
    )


def parse_dataset_name(dataset_name: str) -> DatasetSpec | None:
    match = DATASET_NAME_PATTERN.match(dataset_name)
    if not match:
        return None
    num_classes = match.group("num_classes")
    return DatasetSpec(
        method_name=match.group("method_name"),
        num_samples=int(match.group("num_samples")),
        seq_len=int(match.group("seq_len")),
        num_classes=int(num_classes) if num_classes is not None else None,
        depth_tag=match.group("depth_tag"),
        difficulty=match.group("difficulty"),
    )


def extract_dataset_name(file_path: str, prefix: str = "probing_data_") -> str | None:
    stem = Path(file_path).stem
    if not stem.startswith(prefix):
        return None
    return stem[len(prefix) :]


def build_hiddenstate_file(
    output_dir: Path,
    spec: DatasetSpec | None,
    model_alias: str,
    feature_type: str,
) -> Path:
    suffix = "" if feature_type == "residual" else f"_{feature_type}"
    if spec is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"acts_{model_alias}{suffix}.npz"

    out = output_dir / spec.method_name / str(spec.seq_len) / str(spec.num_samples)
    if spec.num_classes is not None:
        out = out / f"{spec.num_classes}c"
    if spec.depth_tag:
        out = out / spec.depth_tag
    if spec.difficulty:
        out = out / spec.difficulty
    out.mkdir(parents=True, exist_ok=True)
    return out / f"acts_{model_alias}{suffix}.npz"
