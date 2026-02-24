#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render 1x4 grid videos from Metaworld multi-view parquet dataset.

Grid order:
1) third-person RGB
2) third-person GT segmentation mask
3) gripper RGB
4) gripper GT segmentation mask
"""

from __future__ import annotations

import argparse
import colorsys
import io
import os
import re
from pathlib import Path
from typing import Dict, Optional

import imageio.v2 as imageio
import numpy as np
from datasets import load_dataset


def _sanitize(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"


def _default_data_dir() -> Optional[Path]:
    root = os.environ.get("JEPAWM_DSET")
    if not root:
        return None
    return Path(root) / "Metaworld_multiview" / "data"


def _default_output_dir() -> Path:
    root = Path(os.environ.get("JEPAWM_HOME", "."))
    return root / "outputs" / "metaworld_multiview_grid"


def _decode_video(video_obj) -> np.ndarray:
    if isinstance(video_obj, dict):
        if video_obj.get("bytes") is not None:
            reader = imageio.get_reader(io.BytesIO(video_obj["bytes"]), format="mp4")
            frames = [frame for frame in reader]
            reader.close()
            return np.stack(frames, axis=0)
        if video_obj.get("path"):
            reader = imageio.get_reader(video_obj["path"], format="mp4")
            frames = [frame for frame in reader]
            reader.close()
            return np.stack(frames, axis=0)
        raise ValueError("Video dict has neither bytes nor path.")

    # torchcodec VideoDecoder fallback
    frames = []
    for i in range(len(video_obj)):
        frame = video_obj[i]
        frames.append(frame.data.permute(1, 2, 0).numpy())
    return np.stack(frames, axis=0)


def _decode_mask(mask_bytes: bytes) -> np.ndarray:
    with np.load(io.BytesIO(mask_bytes)) as npz:
        if "mask" in npz:
            arr = npz["mask"]
        else:
            # Fallback for unexpected key names
            arr = npz[npz.files[0]]
    return arr.astype(np.int32)


def _build_label_palette(labels: np.ndarray) -> Dict[int, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    unique = sorted(int(x) for x in np.unique(labels).tolist())
    palette: Dict[int, np.ndarray] = {0: np.array([0, 0, 0], dtype=np.uint8)}
    nonzero = [x for x in unique if x != 0]
    if not nonzero:
        return palette

    # Use a low-discrepancy hue sequence for high contrast among nearby labels.
    golden = 0.6180339887498949
    for i, label in enumerate(nonzero):
        hue = (0.13 + i * golden) % 1.0
        sat = 0.95 if (i % 2 == 0) else 0.80
        val = 0.98 if (i % 3 != 0) else 0.82
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette[label] = np.array(
            [int(round(255 * r)), int(round(255 * g)), int(round(255 * b))], dtype=np.uint8
        )
    return palette


def _overlay_mask_boundaries(rgb: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    boundary = np.zeros(mask_hw.shape, dtype=bool)
    boundary[1:, :] |= mask_hw[1:, :] != mask_hw[:-1, :]
    boundary[:-1, :] |= mask_hw[:-1, :] != mask_hw[1:, :]
    boundary[:, 1:] |= mask_hw[:, 1:] != mask_hw[:, :-1]
    boundary[:, :-1] |= mask_hw[:, :-1] != mask_hw[:, 1:]
    out[boundary] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def _mask_to_rgb(mask_hw: np.ndarray, palette: Dict[int, np.ndarray], draw_boundaries: bool) -> np.ndarray:
    flat = mask_hw.reshape(-1)
    unique_vals, inverse = np.unique(flat, return_inverse=True)
    local_palette = np.zeros((len(unique_vals), 3), dtype=np.uint8)
    for i, val in enumerate(unique_vals.tolist()):
        local_palette[i] = palette.get(int(val), np.array([255, 0, 255], dtype=np.uint8))
    rgb = local_palette[inverse].reshape(mask_hw.shape[0], mask_hw.shape[1], 3)
    if draw_boundaries:
        rgb = _overlay_mask_boundaries(rgb, mask_hw)
    return rgb


def _resize_like(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    in_h, in_w = img.shape[:2]
    y_idx = np.linspace(0, in_h - 1, h).astype(np.int32)
    x_idx = np.linspace(0, in_w - 1, w).astype(np.int32)
    return img[y_idx][:, x_idx].astype(np.uint8)


def _render_one(
    row: dict,
    out_path: Path,
    fps: int,
    max_frames: Optional[int],
    draw_boundaries: bool,
) -> int:
    third_rgb = _decode_video(row["video_third"])
    gripper_rgb = _decode_video(row["video_gripper"])
    third_mask = _decode_mask(row["mask_third"])
    gripper_mask = _decode_mask(row["mask_gripper"])

    n = min(len(third_rgb), len(gripper_rgb), len(third_mask), len(gripper_mask))
    if max_frames is not None and max_frames > 0:
        n = min(n, int(max_frames))
    if n <= 0:
        raise ValueError("No frames available to render.")

    label_palette = _build_label_palette(
        np.concatenate([third_mask.reshape(-1), gripper_mask.reshape(-1)], axis=0)
    )

    h, w = third_rgb[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out_path), fps=fps, macro_block_size=1) as writer:
        for t in range(n):
            rgb_t = third_rgb[t].astype(np.uint8)
            rgb_g = _resize_like(gripper_rgb[t].astype(np.uint8), h, w)
            mask_t = _mask_to_rgb(third_mask[t], palette=label_palette, draw_boundaries=draw_boundaries)
            mask_g = _resize_like(
                _mask_to_rgb(gripper_mask[t], palette=label_palette, draw_boundaries=draw_boundaries), h, w
            )
            frame = np.concatenate([rgb_t, mask_t, rgb_g, mask_g], axis=1)
            writer.append_data(frame)
    return n


def main():
    parser = argparse.ArgumentParser(description="Render 1x4 grid videos from Metaworld multi-view parquet.")
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--index", type=int, default=0, help="Start rollout index.")
    parser.add_argument("--count", type=int, default=1, help="How many rollouts to render.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-mask-boundary",
        action="store_true",
        help="Disable white boundaries between mask regions.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/metaworld_multiview_hf_cache"))
    args = parser.parse_args()

    if args.data_dir is None:
        raise ValueError("--data-dir is required (or set JEPAWM_DSET).")
    ds = load_dataset("parquet", data_dir=str(args.data_dir), split="train", cache_dir=str(args.cache_dir))
    total = len(ds)
    if total == 0:
        raise ValueError(f"No rollouts found: {args.data_dir}")

    start = int(args.index)
    end = min(total, start + int(args.count))
    if start < 0 or start >= total:
        raise IndexError(f"index={start} out of range [0, {total - 1}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {total} rollouts from {args.data_dir}")
    for idx in range(start, end):
        row = ds[idx]
        task = _sanitize(row.get("task", "unknown_task"))
        episode = int(row.get("episode", -1))
        out = args.output_dir / f"mw_mv_grid_idx{idx:05d}_{task}_ep{episode}.mp4"
        n = _render_one(
            row,
            out,
            fps=args.fps,
            max_frames=args.max_frames,
            draw_boundaries=not args.no_mask_boundary,
        )
        print(f"[ok] idx={idx} frames={n} -> {out}")


if __name__ == "__main__":
    main()
