"""Build an episode index for DROID by scanning H5 episodes and extracting minimal metadata.

Example:
python tools/build_droid_episode_index.py \
  --droid_root /path/to/droid \
  --output_path outputs/episode_index.jsonl \
  --require_im128 \
  --rgb_key observation/camera/image/varied_camera_1_left_image \
  --action_keys action/cartesian_velocity,action/gripper_velocity
"""

import argparse
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm


@dataclass
class EpisodeConfig:
    rgb_key: str
    action_keys: List[str]
    min_frames: int
    min_action_mag: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan DROID root and build an episode index.")
    parser.add_argument("--droid_root", type=Path, required=True, help="Root directory of DROID.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/episode_index.jsonl"),
        help="Where to write the JSONL episode index.",
    )
    parser.add_argument(
        "--rgb_key",
        type=str,
        default="observation/camera/image/varied_camera_1_left_image",
        help="RGB dataset key inside each H5.",
    )
    parser.add_argument(
        "--action_keys",
        type=str,
        default="action/cartesian_velocity,action/gripper_velocity",
        help="Comma-separated list of action dataset keys.",
    )
    parser.add_argument(
        "--require_im128",
        action="store_true",
        default=True,
        help="If set, only inspect files named trajectory_im128.h5 (default behavior).",
    )
    parser.add_argument(
        "--no-require_im128",
        dest="require_im128",
        action="store_false",
        help="Allow any .h5 episode filename instead of only trajectory_im128.h5.",
    )
    parser.add_argument("--min_frames", type=int, default=64, help="Minimum frames required.")
    parser.add_argument(
        "--min_action_mag",
        type=float,
        default=1e-3,
        help="Minimum average action magnitude to keep an episode.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of files to inspect (useful for debugging).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when subsampling files with --max_files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes (>=1). Sequential if set to 1.",
    )
    parser.add_argument(
        "--write_parquet",
        action="store_true",
        help="If set, also write outputs/episode_index.parquet.",
    )
    return parser.parse_args()


def parse_action_keys(action_keys: str) -> List[str]:
    keys = [k.strip() for k in action_keys.split(",") if k.strip()]
    if not keys:
        raise ValueError("At least one action key must be provided.")
    return keys


def discover_episode_files(root: Path, require_im128: bool) -> List[Path]:
    """Recursively find candidate H5 files under root."""
    candidates: List[Path] = []
    for path in root.rglob("*.h5"):
        if any(part.lower() == "recordings" for part in path.parts):
            continue
        if require_im128 and path.name != "trajectory_im128.h5":
            continue
        candidates.append(path)
    # Deduplicate while preserving order.
    return list(dict.fromkeys(candidates))


def sample_files(files: List[Path], max_files: Optional[int], seed: int) -> List[Path]:
    if max_files is None or len(files) <= max_files:
        return files
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(files), size=max_files, replace=False)
    return [files[i] for i in sorted(indices)]


def parse_lab_outcome(path: Path) -> Tuple[str, str]:
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        lowered = part.lower()
        if lowered in ("success", "failure"):
            if idx == 0:
                break
            return parts[idx - 1], lowered
    return "UNKNOWN", "UNKNOWN"


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D action array, got shape {arr.shape}")


def validate_rgb(f: h5py.File, rgb_key: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if rgb_key not in f:
        return None, "missing_rgb"
    arr = f[rgb_key]
    if not isinstance(arr, h5py.Dataset):
        return None, "missing_rgb"
    if len(arr.shape) != 4 or arr.shape[-1] != 3:
        return None, "invalid_rgb_shape"
    if arr.dtype != np.uint8:
        return None, "invalid_rgb_dtype"
    return arr, None


def validate_actions(
    f: h5py.File, action_keys: Sequence[str], target_t: int
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str], Dict[str, str]]:
    """Return (actions_concat, alignment, reject_reason, action_dtypes)."""
    arrays: List[np.ndarray] = []
    lengths: List[int] = []
    dtypes: Dict[str, str] = {}

    for key in action_keys:
        if key not in f:
            return None, None, "missing_action", dtypes
        ds = f[key]
        if not isinstance(ds, h5py.Dataset):
            return None, None, "missing_action", dtypes
        arr = np.asarray(ds[...], dtype=np.float32)
        try:
            arr = _ensure_2d(arr)
        except ValueError:
            return None, None, "action_shape", dtypes
        arrays.append(arr)
        lengths.append(arr.shape[0])
        dtypes[key] = str(ds.dtype)

    unique_lengths = set(lengths)
    if len(unique_lengths) != 1:
        return None, None, "action_mismatch", dtypes

    action_len = unique_lengths.pop()
    if action_len == target_t:
        alignment = "T"
    elif action_len == target_t - 1:
        alignment = "T-1"
    else:
        return None, None, "action_mismatch", dtypes

    actions = np.concatenate(arrays, axis=1)
    return actions, alignment, None, dtypes


def compute_motion_score(actions: np.ndarray) -> float:
    norms = np.linalg.norm(actions, axis=1)
    return float(norms.mean()) if norms.size else 0.0


def process_episode(path: Path, config: EpisodeConfig) -> Dict[str, object]:
    lab, outcome = parse_lab_outcome(path)
    result: Dict[str, object] = {
        "path": str(path),
        "lab": lab,
        "outcome": outcome,
        "usable": False,
        "reject_reason": None,
        "T": None,
        "rgb_key": config.rgb_key,
        "action_keys": config.action_keys,
        "action_dim": None,
        "action_alignment": None,
        "avg_action_mag": None,
        "dtype_info": None,
        "note": None,
    }

    try:
        with h5py.File(path, "r") as f:
            rgb_ds, rgb_error = validate_rgb(f, config.rgb_key)
            if rgb_error:
                result["reject_reason"] = rgb_error
                return result

            rgb_shape = tuple(int(d) for d in rgb_ds.shape)
            t = rgb_shape[0]
            result["T"] = t

            if t < config.min_frames:
                result["reject_reason"] = "too_short"
                return result

            actions, alignment, action_error, action_dtypes = validate_actions(
                f, config.action_keys, t
            )
            if action_error:
                result["reject_reason"] = action_error
                return result

            result["action_alignment"] = alignment
            result["action_dim"] = int(actions.shape[1])

            avg_action_mag = compute_motion_score(actions)
            result["avg_action_mag"] = avg_action_mag

            if avg_action_mag < config.min_action_mag:
                result["reject_reason"] = "too_static"
                return result

            result["usable"] = True
            result["dtype_info"] = {
                "rgb": str(rgb_ds.dtype),
                "actions": action_dtypes,
            }
            if alignment == "T-1":
                result["note"] = "actions length is T-1"
            return result
    except Exception as exc:  # noqa: BLE001
        result["reject_reason"] = f"open_error: {exc}"
        return result


def compute_summary(records: List[Dict[str, object]]) -> Dict[str, object]:
    reject_counts: Counter = Counter()
    lab_outcome_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    usable_ts: List[int] = []
    usable_motion: List[float] = []

    for rec in records:
        if rec["usable"]:
            lab_outcome_counts[rec["lab"]][rec["outcome"]] += 1
            if rec["T"] is not None:
                usable_ts.append(int(rec["T"]))
            if rec["avg_action_mag"] is not None:
                usable_motion.append(float(rec["avg_action_mag"]))
        else:
            reject_counts[rec["reject_reason"] or "unknown"] += 1

    def stats(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        arr = np.array(values, dtype=float)
        return {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }

    summary = {
        "total": len(records),
        "usable": sum(1 for r in records if r["usable"]),
        "reject_counts": dict(reject_counts),
        "usable_by_lab_outcome": {lab: dict(outcomes) for lab, outcomes in lab_outcome_counts.items()},
        "T_stats": stats(usable_ts),
        "avg_action_mag_stats": stats(usable_motion),
    }
    return summary


def _process_episode_wrapper(args) -> Dict[str, object]:
    path, config = args
    return process_episode(path, config)


def main() -> None:
    args = parse_args()
    action_keys = parse_action_keys(args.action_keys)

    files = discover_episode_files(args.droid_root, args.require_im128)
    if not files:
        raise SystemExit("No candidate H5 files found under the provided root.")

    files = sample_files(files, args.max_files, args.seed)

    config = EpisodeConfig(
        rgb_key=args.rgb_key,
        action_keys=action_keys,
        min_frames=args.min_frames,
        min_action_mag=args.min_action_mag,
    )

    results: List[Dict[str, object]] = []
    if args.num_workers > 1:
        work_items = [(p, config) for p in files]
        with mp.Pool(processes=args.num_workers) as pool:
            for rec in tqdm(
                pool.imap_unordered(_process_episode_wrapper, work_items),
                total=len(files),
                desc="Processing episodes",
            ):
                results.append(rec)
    else:
        for path in tqdm(files, desc="Processing episodes"):
            rec = process_episode(path, config)
            results.append(rec)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote episode index to {args.output_path}")

    summary = compute_summary(results)
    summary_path = Path("outputs/episode_index_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")

    if args.write_parquet:
        try:
            import pandas as pd

            df = pd.DataFrame(results)
            parquet_path = args.output_path.with_suffix(".parquet")
            df.to_parquet(parquet_path, index=False)
            print(f"Wrote parquet to {parquet_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to write parquet: {exc}")


if __name__ == "__main__":
    main()
