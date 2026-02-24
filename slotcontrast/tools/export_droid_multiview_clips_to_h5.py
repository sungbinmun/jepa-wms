import argparse
import json
from pathlib import Path
from typing import List

import h5py
import numpy as np
import tqdm


def sanitize(name: str) -> str:
    return name.replace("/", "__")


def list_action_datasets(f: h5py.File, prefix: str = "action/") -> List[str]:
    out: List[str] = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.startswith(prefix):
            out.append(name)

    f.visititems(_visit)
    out.sort()
    return out


def align_T(arr: np.ndarray, T: int) -> np.ndarray:
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == T:
        return arr
    if arr.shape[0] > T:
        return arr[:T]
    pad_n = T - arr.shape[0]
    pad = np.repeat(arr[-1:], pad_n, axis=0)
    return np.concatenate([arr, pad], axis=0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="jsonl manifest produced by sampler")
    ap.add_argument("--output-h5", type=Path, required=True, help="output HDF5 file")
    ap.add_argument("--store-all-actions", action="store_true",
                    help="Store all action/* datasets. If off, use --action-keys.")
    ap.add_argument("--action-keys", type=str, nargs="*", default=[],
                    help="Explicit action dataset keys to store (e.g. action/abs_pos action/gripper_position).")
    ap.add_argument("--extra-keys", type=str, nargs="*", default=[],
                    help="Additional dataset keys to store (timestamps etc.).")
    ap.add_argument("--compression", type=str, default="gzip", help="HDF5 compression (gzip/lzf/None)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-missing", action="store_true", help="Skip clips if any key is missing.")
    return ap.parse_args()


def main():
    args = parse_args()
    records = []
    with args.manifest.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if args.limit is not None:
        records = records[: args.limit]

    comp = None if args.compression.lower() in ("none", "") else args.compression
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output_h5, "w") as out_f:
        root = out_f.create_group("clips")

        for rec in tqdm.tqdm(records, desc="Exporting"):
            clip_id = rec["clip_id"]
            clip_key = sanitize(clip_id)

            ep_path = rec["episode_path"]
            start = int(rec["start"])
            end = int(rec["end"])

            view_keys = rec.get("view_keys", None)
            if view_keys is None:
                vk = rec.get("view_key", None)
                if vk is None:
                    raise KeyError("Manifest record missing view_keys/view_key")
                view_keys = [vk]

            try:
                with h5py.File(ep_path, "r") as in_f:
                    missing = [k for k in view_keys if k not in in_f]
                    if missing:
                        if args.skip_missing:
                            continue
                        raise KeyError(f"Missing view keys in {ep_path}: {missing}")

                    frames_by_view = {}
                    lengths = []
                    for k in view_keys:
                        arr = in_f[k][start:end]
                        if arr.ndim != 4:
                            raise ValueError(f"Expected frames (T,H,W,C) for {k}, got {arr.shape}")
                        frames_by_view[k] = arr
                        lengths.append(arr.shape[0])

                    T = int(min(lengths))
                    for k in list(frames_by_view.keys()):
                        frames_by_view[k] = frames_by_view[k][:T]

                    g = root.create_group(clip_key)
                    g.attrs["clip_id"] = clip_id
                    g.attrs["episode_path"] = ep_path
                    g.attrs["lab"] = rec.get("lab", "")
                    g.attrs["outcome"] = rec.get("outcome", "")
                    g.attrs["start"] = start
                    g.attrs["end"] = end
                    g.attrs["clip_length"] = int(rec.get("clip_length", end - start))
                    g.attrs["clip_stride"] = int(rec.get("clip_stride", -1))

                    fg = g.create_group("frames")
                    for k, arr in frames_by_view.items():
                        fg.create_dataset(sanitize(k), data=arr, compression=comp, chunks=True)

                    action_keys = list_action_datasets(in_f) if args.store_all_actions else list(args.action_keys)
                    ag = g.create_group("actions")
                    for k in action_keys:
                        if k not in in_f:
                            continue
                        arr = np.array(in_f[k][start:end])
                        arr = align_T(arr, T)
                        ag.create_dataset(sanitize(k), data=arr, compression=None)

                    eg = g.create_group("extra")
                    for k in args.extra_keys:
                        if k not in in_f:
                            continue
                        arr = np.array(in_f[k][start:end])
                        if arr.ndim >= 1 and arr.shape[0] != T:
                            arr = align_T(arr, T)
                        eg.create_dataset(sanitize(k), data=arr, compression=None)

            except Exception as e:
                if args.skip_missing:
                    continue
                raise e

        out_f.attrs["format"] = "droid_multiview_clips_v1"
        out_f.attrs["num_clips"] = len(root.keys())

    print(f"[ok] wrote {args.output_h5}")


if __name__ == "__main__":
    main()