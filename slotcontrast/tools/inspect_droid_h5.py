"""Inspect DROID HDF5 episodes to discover schema and likely RGB/action keys.

Supports either scanning a local directory of example H5 files or reading a manifest of paths.
The script prints per-file candidate datasets and emits an aggregate JSON report for later use.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


@dataclass
class DatasetInfo:
    key: str
    shape: Tuple[int, ...]
    dtype: np.dtype


@dataclass
class FileSummary:
    path: Path
    rgb_candidates: List[Dict[str, object]]
    action_candidates: List[Dict[str, object]]
    best_rgb: Optional[Dict[str, object]]
    temporal_length: Optional[int]
    spatial_hw: Optional[Tuple[int, int]]
    num_datasets: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect DROID HDF5 files for schema discovery.")
    parser.add_argument(
        "--examples_dir",
        type=Path,
        default=Path("droid/2023-04-20"),
        help="Directory containing example HDF5 episodes (scanned recursively).",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=None,
        help="Optional manifest JSON of H5 paths; if provided, overrides --examples_dir.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Maximum number of files to inspect after deduplication.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling files.")
    parser.add_argument(
        "--output_report",
        type=Path,
        default=Path("outputs/h5_schema_report.json"),
        help="Where to write the aggregated JSON report.",
    )
    parser.add_argument(
        "--print_tree",
        action="store_true",
        help="If set, print dataset keys/shapes/dtypes for each inspected file.",
    )
    parser.add_argument(
        "--max_datasets_per_file",
        type=int,
        default=200,
        help="Maximum number of dataset entries to print per file when --print_tree is set.",
    )
    parser.add_argument(
        "--path_field",
        type=str,
        default="path",
        help="Dictionary field to read file paths from when using a manifest.",
    )
    return parser.parse_args()


def load_manifest_paths(manifest_path: Path, path_field: str) -> List[str]:
    """Load list of H5 paths from a manifest JSON (list of dicts or list of strings)."""
    text = manifest_path.read_text().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return extract_paths_from_json(data, path_field)
    except json.JSONDecodeError:
        # Try JSONL as a fallback.
        paths: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                paths.extend(extract_paths_from_json(obj, path_field))
            except json.JSONDecodeError:
                paths.append(line)
        return paths


def extract_paths_from_json(obj: object, path_field: str) -> List[str]:
    if isinstance(obj, list):
        return _extract_paths_from_list(obj, path_field)
    if isinstance(obj, dict):
        if path_field in obj and isinstance(obj[path_field], str):
            return [obj[path_field]]
        for key in ("items", "episodes", "data"):
            if key in obj:
                return extract_paths_from_json(obj[key], path_field)
        raise ValueError("Manifest dict missing expected path field.")
    if isinstance(obj, str):
        return [obj]
    raise ValueError(f"Unsupported manifest format: {type(obj)}")


def _extract_paths_from_list(seq: Sequence[object], path_field: str) -> List[str]:
    paths: List[str] = []
    for item in seq:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            if path_field in item:
                paths.append(str(item[path_field]))
            else:
                raise ValueError(f"Manifest entry missing '{path_field}': {item}")
        else:
            raise ValueError(f"Unsupported manifest entry type: {type(item)}")
    return paths


def gather_candidate_paths(
    examples_dir: Path,
    manifest_path: Optional[Path],
    num_examples: int,
    seed: int,
    path_field: str,
) -> List[Path]:
    if manifest_path:
        paths = load_manifest_paths(manifest_path, path_field)
    else:
        paths = [str(p) for p in examples_dir.rglob("*.h5")]

    # Deduplicate while preserving order.
    unique = list(dict.fromkeys(paths))
    if not unique:
        return []

    rng = np.random.default_rng(seed)
    if len(unique) > num_examples:
        indices = rng.choice(len(unique), size=num_examples, replace=False)
        unique = [unique[i] for i in sorted(indices)]

    return [Path(p) for p in unique]


def collect_dataset_info(h5_path: Path) -> List[DatasetInfo]:
    datasets: List[DatasetInfo] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                shape = tuple(int(dim) for dim in obj.shape)
            except Exception:
                shape = tuple(obj.shape)
            datasets.append(DatasetInfo(key=name, shape=shape, dtype=obj.dtype))

    with h5py.File(h5_path, "r") as f:
        f.visititems(visitor)
    return datasets


def rgb_score(shape: Tuple[int, ...], dtype: np.dtype) -> Optional[float]:
    if dtype != np.uint8:
        return None
    if len(shape) != 4 or shape[-1] != 3:
        return None
    t, h, w, _ = shape
    target_hw = 128.0
    mean_hw = (h + w) / 2.0
    score = 0.0
    score += 1.0  # baseline for matching rank/dtype
    score -= abs(mean_hw - target_hw) / target_hw
    score -= abs(h - w) / max(max(h, w), 1)
    score += min(t, 64) / 64.0  # prefer clips with some temporal length
    return score


def action_score(
    shape: Tuple[int, ...], dtype: np.dtype, candidate_rgb_ts: Iterable[int]
) -> Optional[float]:
    if len(shape) != 2:
        return None
    if not np.issubdtype(dtype, np.floating):
        return None
    t, dim = shape
    score = 0.0
    if dim <= 32:
        score += 1.0
    else:
        score -= (dim - 32) / 64.0

    rgb_ts = list(candidate_rgb_ts)
    if rgb_ts:
        closest = min(rgb_ts, key=lambda x: abs(x - t))
        diff = abs(closest - t)
        if diff == 0:
            score += 2.0
        elif diff == 1:
            score += 1.5
        else:
            score += max(0.0, 1.0 - diff / max(closest, t, 1))
    else:
        score += 0.3  # weak prior when no RGB temporal info

    score += min(t, 128) / 256.0  # mild boost for longer sequences
    return score


def rank_candidates(
    datasets: List[DatasetInfo],
    max_rgb: int = 3,
    max_action: int = 3,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rgb_candidates: List[Dict[str, object]] = []
    for ds in datasets:
        score = rgb_score(ds.shape, ds.dtype)
        if score is None:
            continue
        rgb_candidates.append(
            {
                "key": ds.key,
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
                "score": round(float(score), 4),
            }
        )
    # Use all RGB temporal lengths to score actions.
    rgb_temporal_lengths = [c["shape"][0] for c in rgb_candidates]

    action_candidates: List[Dict[str, object]] = []
    for ds in datasets:
        score = action_score(ds.shape, ds.dtype, rgb_temporal_lengths)
        if score is None:
            continue
        action_candidates.append(
            {
                "key": ds.key,
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
                "score": round(float(score), 4),
            }
        )

    rgb_candidates = sorted(
        rgb_candidates, key=lambda x: (-x["score"], x["key"])
    )[:max_rgb]
    action_candidates = sorted(
        action_candidates, key=lambda x: (-x["score"], x["key"])
    )[:max_action]
    return rgb_candidates, action_candidates


def print_file_summary(
    path: Path,
    rgb_candidates: List[Dict[str, object]],
    action_candidates: List[Dict[str, object]],
    datasets: List[DatasetInfo],
    print_tree: bool,
    max_entries: int,
) -> None:
    print(f"\nFile: {path}")
    if rgb_candidates:
        print("  RGB candidates (top 3):")
        for c in rgb_candidates:
            print(f"    - {c['key']}: shape={c['shape']} dtype={c['dtype']} score={c['score']}")
    else:
        print("  RGB candidates: none")

    if action_candidates:
        print("  Action candidates (top 3):")
        for c in action_candidates:
            print(f"    - {c['key']}: shape={c['shape']} dtype={c['dtype']} score={c['score']}")
    else:
        print("  Action candidates: none")

    if print_tree:
        print(f"  Datasets (up to {max_entries}):")
        for ds in sorted(datasets, key=lambda d: d.key)[:max_entries]:
            print(f"    * {ds.key}: shape={list(ds.shape)} dtype={ds.dtype}")


def to_file_summary(
    path: Path,
    rgb_candidates: List[Dict[str, object]],
    action_candidates: List[Dict[str, object]],
    datasets: List[DatasetInfo],
) -> FileSummary:
    best_rgb = rgb_candidates[0] if rgb_candidates else None
    temporal_length = best_rgb["shape"][0] if best_rgb else None
    spatial_hw = (
        (best_rgb["shape"][1], best_rgb["shape"][2]) if best_rgb and len(best_rgb["shape"]) >= 3 else None
    )
    return FileSummary(
        path=path,
        rgb_candidates=rgb_candidates,
        action_candidates=action_candidates,
        best_rgb=best_rgb,
        temporal_length=temporal_length,
        spatial_hw=spatial_hw,
        num_datasets=len(datasets),
    )


def write_report(
    output_path: Path,
    summaries: List[FileSummary],
    rgb_freq: Counter,
    action_freq: Counter,
    failures: List[Dict[str, str]],
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "inspected_files": [str(s.path) for s in summaries],
        "rgb_candidates_by_frequency": [
            {"key": k, "count": v} for k, v in rgb_freq.most_common()
        ],
        "action_candidates_by_frequency": [
            {"key": k, "count": v} for k, v in action_freq.most_common()
        ],
        "per_file_summary": [
            {
                "path": str(s.path),
                "rgb_candidates": s.rgb_candidates,
                "action_candidates": s.action_candidates,
                "best_rgb": s.best_rgb,
                "temporal_length": s.temporal_length,
                "spatial_hw": list(s.spatial_hw) if s.spatial_hw else None,
                "num_datasets": s.num_datasets,
            }
            for s in summaries
        ],
        "failures": failures,
        "args": {
            "examples_dir": str(args.examples_dir),
            "manifest_path": str(args.manifest_path) if args.manifest_path else None,
            "num_examples": args.num_examples,
            "seed": args.seed,
            "path_field": args.path_field,
        },
    }
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {output_path}")


def main() -> None:
    args = parse_args()
    candidate_paths = gather_candidate_paths(
        examples_dir=args.examples_dir,
        manifest_path=args.manifest_path,
        num_examples=args.num_examples,
        seed=args.seed,
        path_field=args.path_field,
    )

    if not candidate_paths:
        raise SystemExit("No candidate H5 files found.")

    failures: List[Dict[str, str]] = []
    summaries: List[FileSummary] = []
    rgb_freq: Counter = Counter()
    action_freq: Counter = Counter()

    for path in candidate_paths:
        if not path.exists():
            reason = "missing"
            failures.append({"path": str(path), "reason": reason})
            print(f"[warn] Skipping missing file: {path}")
            continue
        try:
            datasets = collect_dataset_info(path)
            rgb_candidates, action_candidates = rank_candidates(datasets)
            print_file_summary(
                path,
                rgb_candidates,
                action_candidates,
                datasets,
                args.print_tree,
                args.max_datasets_per_file,
            )
            summary = to_file_summary(path, rgb_candidates, action_candidates, datasets)
            summaries.append(summary)
            for c in rgb_candidates:
                rgb_freq[c["key"]] += 1
            for c in action_candidates:
                action_freq[c["key"]] += 1
        except Exception as exc:  # noqa: BLE001
            reason = f"error: {exc}"
            failures.append({"path": str(path), "reason": reason})
            print(f"[warn] Failed to inspect {path}: {exc}")

    write_report(args.output_report, summaries, rgb_freq, action_freq, failures, args)


if __name__ == "__main__":
    main()
