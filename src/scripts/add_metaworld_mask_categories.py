#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Add category masks (bg/fg/agent) to existing Metaworld multiview parquet shards.

This script does not re-render MuJoCo frames. It reads existing instance-id masks:
- mask_third
- mask_gripper

Then it writes:
- mask_cat_third (0=bg, 1=fg, 2=agent)
- mask_cat_gripper (0=bg, 1=fg, 2=agent)

By default, shards are updated in place.
"""

from __future__ import annotations

import argparse
import io
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import preprocess_metaworld_multiview as pm


def _default_data_dir() -> Optional[Path]:
    root = os.environ.get("JEPAWM_DSET")
    if not root:
        return None
    return Path(root) / "Metaworld_multiview_full" / "data"


def _decode_npz_mask(data: bytes) -> np.ndarray:
    with np.load(io.BytesIO(data)) as npz:
        key = "mask" if "mask" in npz.files else npz.files[0]
        arr = npz[key]
    return arr


def _encode_npz_mask(data: np.ndarray) -> bytes:
    arr = data.astype(np.int32, copy=False)
    buf = io.BytesIO()
    np.savez_compressed(buf, mask=arr)
    return buf.getvalue()


def _resolve_category_sets(
    task: str, cache: Dict[str, Tuple[set[int], set[int]]]
) -> Tuple[set[int], set[int]]:
    out = cache.get(task)
    if out is not None:
        return out

    env = pm._make_env(task)
    try:
        out = pm._build_category_id_sets(env.model)
    finally:
        env.close()
    cache[task] = out
    return out


def _shard_index(path: Path) -> int:
    m = re.match(r"train-(\d+)-of-(\d+)\.parquet$", path.name)
    if m is None:
        return -1
    return int(m.group(1))


def _process_shard(
    shard_path: Path,
    dst_path: Path,
    cache_dir: Path,
    category_sets: Dict[str, Tuple[set[int], set[int]]],
    overwrite: bool,
    quiet: bool,
):
    ds = load_dataset(
        "parquet",
        data_files=str(shard_path),
        split="train",
        cache_dir=str(cache_dir),
    )

    has_cat_third = "mask_cat_third" in ds.column_names
    has_cat_gripper = "mask_cat_gripper" in ds.column_names
    if has_cat_third and has_cat_gripper and not overwrite:
        if not quiet:
            print(f"[skip] already has mask_cat columns: {shard_path.name}")
        return

    tasks = ds["task"]
    mask_third = ds["mask_third"]
    mask_gripper = ds["mask_gripper"]

    cat_third_out = []
    cat_gripper_out = []

    pbar = tqdm(range(len(ds)), desc=shard_path.name, leave=False, disable=quiet)
    for i in pbar:
        task = str(tasks[i])
        bg_ids, agent_ids = _resolve_category_sets(task, category_sets)

        third = _decode_npz_mask(mask_third[i])
        gripper = _decode_npz_mask(mask_gripper[i])
        # Normalize potentially swapped packed layouts from older exports.
        third = pm._normalize_packed_instance_mask(third)
        gripper = pm._normalize_packed_instance_mask(gripper)

        third_cat = pm._instance_to_category_mask(third, bg_ids, agent_ids)
        gripper_cat = pm._instance_to_category_mask(gripper, bg_ids, agent_ids)

        cat_third_out.append(_encode_npz_mask(third_cat))
        cat_gripper_out.append(_encode_npz_mask(gripper_cat))

    if has_cat_third:
        ds = ds.remove_columns("mask_cat_third")
    if has_cat_gripper:
        ds = ds.remove_columns("mask_cat_gripper")

    ds = ds.add_column("mask_cat_third", cat_third_out)
    ds = ds.add_column("mask_cat_gripper", cat_gripper_out)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".tmp.parquet")
    ds.to_parquet(str(tmp_path))
    os.replace(tmp_path, dst_path)

    # Avoid cache growth during long runs.
    ds.cleanup_cache_files()
    if not quiet:
        print(f"[ok] {shard_path.name} -> {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add mask_cat_third/mask_cat_gripper to Metaworld parquet shards."
    )
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If unset, shards are updated in place.",
    )
    parser.add_argument("--pattern", type=str, default="train-*.parquet")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/metaworld_hf_cache"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--start-shard",
        type=int,
        default=None,
        help="Inclusive shard index filter (e.g., 0 for train-00000-...).",
    )
    parser.add_argument(
        "--end-shard",
        type=int,
        default=None,
        help="Exclusive shard index filter.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.data_dir is None:
        raise ValueError("Pass --data-dir or set JEPAWM_DSET.")
    if not args.data_dir.exists():
        raise ValueError(f"Data dir not found: {args.data_dir}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(args.data_dir.glob(args.pattern))
    if args.start_shard is not None or args.end_shard is not None:
        start = 0 if args.start_shard is None else int(args.start_shard)
        end = 10**9 if args.end_shard is None else int(args.end_shard)
        shards = [
            s
            for s in shards
            if (idx := _shard_index(s)) >= 0 and start <= idx < end
        ]
    if args.max_shards is not None:
        shards = shards[: int(args.max_shards)]
    if not shards:
        raise ValueError(f"No shards matched pattern '{args.pattern}' in {args.data_dir}")

    print(f"Found {len(shards)} shards in {args.data_dir}")
    if args.output_dir is None:
        print("Mode: in-place update")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Mode: write to {args.output_dir}")

    category_sets: Dict[str, Tuple[set[int], set[int]]] = {}
    for shard in shards:
        dst = shard if args.output_dir is None else (args.output_dir / shard.name)
        _process_shard(
            shard_path=shard,
            dst_path=dst,
            cache_dir=args.cache_dir,
            category_sets=category_sets,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )

    print("Done.")


if __name__ == "__main__":
    main()
