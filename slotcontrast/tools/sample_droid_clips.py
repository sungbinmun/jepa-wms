"""
Sample DROID trajectories into a clip manifest with stratified lab/outcome sampling.

This version supports heterogeneous camera key naming across labs by selecting
per-episode view keys from candidate lists.

Output: jsonl manifest; each record contains:
- episode_path, start, end, clip_length, clip_stride
- lab, outcome (parsed from path .../<lab>/<success|failure>/...)
- view_keys: [exo_key_selected, hand_key_selected]
- num_frames: min frames across selected views for that episode

Notes:
- clip_stride here means stride between clip *start positions* (not intra-clip subsampling).
- This script only creates a manifest; exporting actual frames/actions is a separate step.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tqdm


@dataclass
class EpisodeEntry:
    path: Path
    lab: str
    outcome: str  # "success" or "failure"
    view_keys: Optional[Tuple[str, str]] = None  # (exo_key, hand_key) chosen for this episode
    num_frames: Optional[int] = None  # min frames across chosen views
    num_clips: Optional[int] = None

    def clip_dict(
        self,
        start: int,
        clip_length: int,
        clip_stride: int,
        clip_id: int,
    ) -> Dict[str, object]:
        assert self.view_keys is not None, "view_keys not set for episode"
        return {
            "clip_id": f"{self.outcome}/{self.lab}/{clip_id:07d}",
            "episode_path": str(self.path),
            "lab": self.lab,
            "outcome": self.outcome,
            "start": int(start),
            "end": int(start + clip_length),
            "clip_length": int(clip_length),
            "clip_stride": int(clip_stride),
            "view_keys": [self.view_keys[0], self.view_keys[1]],
            "num_frames": int(self.num_frames) if self.num_frames is not None else None,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample DROID clips with lab/outcome stratification and per-episode view-key selection."
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest (json/jsonl/plain lines) listing trajectory_im128.h5 paths.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output sampled clip manifest (jsonl).")

    # View selection (per episode)
    p.add_argument(
        "--exo-candidates",
        type=str,
        nargs="+",
        required=True,
        help="Ordered list of exocentric view dataset keys (first existing key is chosen per episode).",
    )
    p.add_argument(
        "--hand-candidates",
        type=str,
        nargs="+",
        required=True,
        help="Ordered list of hand/gripper view dataset keys (first existing key is chosen per episode).",
    )
    p.add_argument(
        "--require-both",
        action="store_true",
        help="If set, skip episodes unless both an exo and hand key can be found.",
    )
    p.add_argument(
        "--strict-key",
        action="store_true",
        help="If set, raise an error when a required key cannot be found (instead of skipping).",
    )

    # Sampling params
    p.add_argument("--clip-length", type=int, default=60, help="Frames per clip (e.g., ~60 for ~4s @15Hz).")
    p.add_argument("--clip-stride", type=int, default=30,
                   help="Stride between clip start positions (frames).")
    p.add_argument("--target-clips", type=int, default=50000, help="Target number of clips to sample.")
    p.add_argument("--failure-ratio", type=float, default=0.2, help="Global fraction of failure clips.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")

    # Behavior knobs
    p.add_argument("--one-clip-per-episode", action="store_true",
                   help="If set, sample at most one clip per episode.")
    p.add_argument("--center-clip", action="store_true",
                   help="If set, choose a center-aligned start (snapped to stride grid).")
    p.add_argument("--shuffle-output", action="store_true",
                   help="Shuffle the output clip list before writing.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on how many manifest entries to process (dry run).")

    return p.parse_args()


def load_manifest(manifest_path: Path) -> List[str]:
    raw_text = manifest_path.read_text().strip()
    if not raw_text:
        return []
    try:
        parsed = json.loads(raw_text)
        return _extract_paths_from_json(parsed)
    except json.JSONDecodeError:
        return _load_jsonl_or_lines(manifest_path)


def _extract_paths_from_json(obj: object) -> List[str]:
    if isinstance(obj, list):
        return _extract_paths_from_list(obj)
    if isinstance(obj, dict):
        for key in ("episodes", "items", "paths", "data"):
            if key in obj:
                return _extract_paths_from_json(obj[key])
        raise ValueError("Could not find list of paths inside manifest dict")
    raise ValueError(f"Unsupported manifest format: {type(obj)}")


def _extract_paths_from_list(seq: Sequence[object]) -> List[str]:
    paths: List[str] = []
    for item in seq:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            for key in ("path", "episode_path", "file", "filepath"):
                if key in item:
                    paths.append(str(item[key]))
                    break
            else:
                raise ValueError(f"Manifest entry missing path-like field: {item}")
        else:
            raise ValueError(f"Unsupported manifest entry type: {type(item)}")
    return paths


def _load_jsonl_or_lines(manifest_path: Path) -> List[str]:
    paths: List[str] = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict):
                paths.extend(_extract_paths_from_list([item]))
            else:
                raise ValueError
        except json.JSONDecodeError:
            paths.append(line)
    return paths


def extract_lab_outcome(path: str) -> Tuple[str, str]:
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        lowered = part.lower()
        if lowered in ("success", "failure"):
            if idx == 0:
                raise ValueError(f"Cannot infer lab from path: {path}")
            return parts[idx - 1], lowered
    raise ValueError(f"Could not find success/failure segment in path: {path}")


def choose_first_existing_key(f: h5py.File, candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in f:
            return k
    return None


def get_episode_views_and_min_frames(
    h5_path: Path,
    exo_candidates: Sequence[str],
    hand_candidates: Sequence[str],
    require_both: bool,
    strict_key: bool,
) -> Tuple[Optional[Tuple[str, str]], Optional[int]]:
    """Return ((exo_key, hand_key), min_frames) or (None, None) if unusable."""
    if not h5_path.is_file():
        print(f"[warn] Missing file, skipping: {h5_path}")
        return None, None

    try:
        with h5py.File(h5_path, "r") as f:
            exo_key = choose_first_existing_key(f, exo_candidates)
            hand_key = choose_first_existing_key(f, hand_candidates)

            if require_both:
                if exo_key is None or hand_key is None:
                    msg = f"[warn] Missing required views in {h5_path} (exo={exo_key}, hand={hand_key})"
                    if strict_key:
                        raise KeyError(msg)
                    print(msg + " (skipping episode)")
                    return None, None
            else:
                # If not require_both, still need at least one view to compute frames;
                # but for this project you probably want both -> recommend --require-both.
                if exo_key is None and hand_key is None:
                    msg = f"[warn] No candidate views found in {h5_path}"
                    if strict_key:
                        raise KeyError(msg)
                    print(msg + " (skipping episode)")
                    return None, None
                # If one missing, we still can't produce 2-view clips; skip for safety.
                if exo_key is None or hand_key is None:
                    msg = f"[warn] One view missing in {h5_path} (exo={exo_key}, hand={hand_key})"
                    if strict_key:
                        raise KeyError(msg)
                    print(msg + " (skipping episode)")
                    return None, None

            # Compute min frames across the two chosen views
            exo_T = int(f[exo_key].shape[0])
            hand_T = int(f[hand_key].shape[0])
            return (exo_key, hand_key), int(min(exo_T, hand_T))

    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to read {h5_path} ({exc}), skipping")
        return None, None


def compute_num_clips(num_frames: int, clip_length: int, clip_stride: int) -> int:
    if num_frames < clip_length:
        return 0
    return (num_frames - clip_length) // clip_stride + 1


def allocate_targets_by_lab(
    entries: Iterable[EpisodeEntry], target: int, rng: np.random.Generator
) -> Dict[str, int]:
    lab_totals: Dict[str, int] = {}
    for e in entries:
        if e.num_clips is None:
            continue
        lab_totals[e.lab] = lab_totals.get(e.lab, 0) + e.num_clips

    total_available = sum(lab_totals.values())
    if total_available == 0 or target == 0:
        return {}

    target = min(target, total_available)
    raw_allocs = {lab: target * clips / total_available for lab, clips in lab_totals.items()}
    allocations = {lab: min(lab_totals[lab], math.floor(val)) for lab, val in raw_allocs.items()}

    remaining = target - sum(allocations.values())
    if remaining > 0:
        fractional = [(lab, raw_allocs[lab] - allocations[lab]) for lab in lab_totals.keys()]
        rng.shuffle(fractional)
        fractional.sort(key=lambda x: x[1], reverse=True)
        for lab, _ in fractional:
            if remaining == 0:
                break
            room = lab_totals[lab] - allocations[lab]
            if room <= 0:
                continue
            allocations[lab] += 1
            remaining -= 1

    return allocations


def allocate_targets_by_episode(
    entries: List[EpisodeEntry], lab_target: int, rng: np.random.Generator
) -> List[Tuple[EpisodeEntry, int]]:
    avail = np.array([e.num_clips for e in entries], dtype=int)
    total = int(avail.sum())
    if total == 0 or lab_target == 0:
        return []

    lab_target = min(lab_target, total)
    raw = lab_target * (avail / total)
    base = np.minimum(avail, np.floor(raw).astype(int))
    remaining = lab_target - int(base.sum())

    if remaining > 0:
        frac = raw - base
        order = np.arange(len(entries))
        rng.shuffle(order)
        order = sorted(order, key=lambda i: frac[i], reverse=True)
        for i in order:
            if remaining == 0:
                break
            if base[i] < avail[i]:
                base[i] += 1
                remaining -= 1

    return [(e, int(c)) for e, c in zip(entries, base) if c > 0]


def sample_offsets(
    num_frames: int, clip_length: int, clip_stride: int, n: int, rng: np.random.Generator
) -> List[int]:
    starts = np.arange(0, num_frames - clip_length + 1, clip_stride, dtype=int)
    if len(starts) <= n:
        rng.shuffle(starts)
        return starts.tolist()
    return rng.choice(starts, size=n, replace=False).tolist()


def center_offset(num_frames: int, clip_length: int, clip_stride: int) -> int:
    max_start = num_frames - clip_length
    desired = max_start / 2.0
    start = int(round(desired / clip_stride) * clip_stride)
    return max(0, min(start, max_start))


def sample_clips_for_outcome(
    entries: List[EpisodeEntry],
    lab_targets: Dict[str, int],
    clip_length: int,
    clip_stride: int,
    rng: np.random.Generator,
    clip_id_offset: int,
    one_clip_per_episode: bool,
    center_clip: bool,
) -> Tuple[List[Dict[str, object]], int]:
    clips: List[Dict[str, object]] = []
    next_id = clip_id_offset

    for lab, lab_target in lab_targets.items():
        lab_entries = [e for e in entries if e.lab == lab and (e.num_clips or 0) > 0]
        episode_allocs = allocate_targets_by_episode(lab_entries, lab_target, rng)

        for e, n in episode_allocs:
            if one_clip_per_episode:
                n = min(n, 1)
            if n <= 0:
                continue

            if center_clip:
                offsets = [center_offset(int(e.num_frames), clip_length, clip_stride)]
            else:
                offsets = sample_offsets(int(e.num_frames), clip_length, clip_stride, n, rng)

            for s in offsets:
                clips.append(e.clip_dict(start=s, clip_length=clip_length, clip_stride=clip_stride, clip_id=next_id))
                next_id += 1

    return clips, next_id


def summarize(entries: List[EpisodeEntry]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for e in entries:
        out.setdefault(e.outcome, {}).setdefault(e.lab, 0)
        out[e.outcome][e.lab] += e.num_clips or 0
    return out


def print_summary(title: str, summary: Dict[str, Dict[str, int]]) -> None:
    print(f"\n{title}")
    for outcome, labs in summary.items():
        total = sum(labs.values())
        print(f"  {outcome}: {total} clips across {len(labs)} labs")
        for lab, count in sorted(labs.items()):
            print(f"    - {lab}: {count}")


def main() -> None:
    args = parse_args()

    if args.clip_length <= 0 or args.clip_stride <= 0:
        raise ValueError("Both --clip-length and --clip-stride must be positive.")
    if not 0 <= args.failure_ratio <= 1:
        raise ValueError("--failure-ratio must be in [0, 1].")

    rng = np.random.default_rng(args.seed)

    manifest_paths = load_manifest(args.manifest)
    manifest_paths = list(dict.fromkeys(manifest_paths))
    if args.limit is not None:
        manifest_paths = manifest_paths[: args.limit]

    entries: List[EpisodeEntry] = []
    for path_str in manifest_paths:
        try:
            lab, outcome = extract_lab_outcome(path_str)
        except ValueError as exc:
            print(f"[warn] {exc}, skipping entry")
            continue
        entries.append(EpisodeEntry(path=Path(path_str), lab=lab, outcome=outcome))

    if not entries:
        raise RuntimeError("No usable manifest entries found.")

    # Scan episodes to choose per-episode view keys and compute available clips
    filtered: List[EpisodeEntry] = []
    for e in tqdm.tqdm(entries, desc="Scanning episodes"):
        views, nf = get_episode_views_and_min_frames(
            e.path,
            exo_candidates=args.exo_candidates,
            hand_candidates=args.hand_candidates,
            require_both=args.require_both,
            strict_key=args.strict_key,
        )
        if views is None or nf is None:
            continue

        e.view_keys = views
        e.num_frames = nf
        e.num_clips = compute_num_clips(nf, args.clip_length, args.clip_stride)
        if e.num_clips and e.num_clips > 0:
            filtered.append(e)

    entries = filtered
    if not entries:
        raise RuntimeError("No episodes with valid frames/views were found (missing keys or too short).")

    summary = summarize(entries)
    print_summary("Available clips after filtering", summary)

    success_entries = [e for e in entries if e.outcome == "success"]
    failure_entries = [e for e in entries if e.outcome == "failure"]

    success_available = sum(e.num_clips or 0 for e in success_entries)
    failure_available = sum(e.num_clips or 0 for e in failure_entries)

    target_failure = min(int(round(args.target_clips * args.failure_ratio)), failure_available)
    target_success = min(args.target_clips - target_failure, success_available)

    shortfall = args.target_clips - (target_success + target_failure)
    if shortfall > 0:
        extra_success = success_available - target_success
        take_success = min(extra_success, shortfall)
        target_success += take_success
        shortfall -= take_success

        if shortfall > 0:
            extra_failure = failure_available - target_failure
            take_failure = min(extra_failure, shortfall)
            target_failure += take_failure
            shortfall -= take_failure

    print(
        f"\nTarget clips: success={target_success}, failure={target_failure} "
        f"(requested total {args.target_clips})"
    )

    failure_lab_targets = allocate_targets_by_lab(failure_entries, target_failure, rng)
    success_lab_targets = allocate_targets_by_lab(success_entries, target_success, rng)

    clips: List[Dict[str, object]] = []
    clip_id = 0

    failure_clips, clip_id = sample_clips_for_outcome(
        failure_entries,
        failure_lab_targets,
        args.clip_length,
        args.clip_stride,
        rng,
        clip_id,
        one_clip_per_episode=args.one_clip_per_episode,
        center_clip=args.center_clip,
    )
    success_clips, clip_id = sample_clips_for_outcome(
        success_entries,
        success_lab_targets,
        args.clip_length,
        args.clip_stride,
        rng,
        clip_id,
        one_clip_per_episode=args.one_clip_per_episode,
        center_clip=args.center_clip,
    )

    clips.extend(failure_clips)
    clips.extend(success_clips)

    if args.shuffle_output:
        rng.shuffle(clips)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in clips:
            json.dump(r, f)
            f.write("\n")

    print(f"\nWrote {len(clips)} clips to {args.output}")


if __name__ == "__main__":
    main()