#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render side-by-side videos of RGB frames and DINOv3 PCA feature maps.

Supported sources:
- Metaworld HF parquet dataset (`$JEPAWM_DSET/Metaworld/data`)
- RoboCasa HDF5 dataset (e.g. `$JEPAWM_DSET/robocasa/combine_all_im256.hdf5`)

Outputs one or more MP4 files per selected source with a 1x2 grid:
left = resized RGB frame, right = PCA-projected DINOv3 patch features.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import timm
from datasets import load_dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


@dataclass
class PCAProjector:
    mean: torch.Tensor  # [C]
    components: torch.Tensor  # [3, C]
    low: torch.Tensor  # [3]
    high: torch.Tensor  # [3]


def _sanitize(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"


def _decode_metaworld_video(video_obj) -> np.ndarray:
    if isinstance(video_obj, dict) and "bytes" in video_obj:
        reader = imageio.get_reader(io.BytesIO(video_obj["bytes"]), format="mp4")
        frames = [frame for frame in reader]
        reader.close()
        return np.stack(frames, axis=0)

    # torchcodec VideoDecoder object.
    frames = []
    for i in range(len(video_obj)):
        frame = video_obj[i]
        frame_np = frame.data.permute(1, 2, 0).numpy()
        frames.append(frame_np)
    return np.stack(frames, axis=0)


def _load_metaworld_frames(data_dir: Path, rollout_idx: int) -> Tuple[np.ndarray, dict]:
    ds = load_dataset("parquet", data_dir=str(data_dir), split="train")
    if len(ds) == 0:
        raise ValueError(f"No rollouts found in Metaworld dataset: {data_dir}")
    if rollout_idx < 0 or rollout_idx >= len(ds):
        raise IndexError(f"rollout_idx={rollout_idx} out of range [0, {len(ds) - 1}]")
    row = ds[rollout_idx]
    frames = _decode_metaworld_video(row["video"])
    task = row.get("task", "unknown_task")
    return frames, {"task": str(task), "rollout_idx": rollout_idx, "num_rollouts": len(ds)}


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def _demo_sort_key(key: str):
        tail = key.split("_")[-1]
        return int(tail) if tail.isdigit() else key

    return sorted(list(data_group.keys()), key=_demo_sort_key)


def _find_robocasa_h5(path: Path, file_index: int) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"RoboCasa path does not exist: {path}")
    candidates = sorted(path.rglob("*.hdf5")) + sorted(path.rglob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under: {path}")
    if file_index < 0 or file_index >= len(candidates):
        raise IndexError(f"robocasa_file_index={file_index} out of range [0, {len(candidates) - 1}]")
    return candidates[file_index]


def _select_camera_key(obs_group: h5py.Group, camera_key: Optional[str]) -> str:
    image_keys = [k for k in obs_group.keys() if k.endswith("_image")]
    if not image_keys:
        raise KeyError("No '*_image' datasets found under demo['obs']")

    if camera_key:
        if camera_key in obs_group:
            return camera_key
        fallback = camera_key + "_image"
        if fallback in obs_group:
            return fallback
        raise KeyError(f"Requested camera key not found: {camera_key}")

    # Default preference: agent view first, then first available.
    for k in image_keys:
        if "agentview" in k:
            return k
    return image_keys[0]


def _get_robocasa_image_keys(
    data_path: Path,
    file_index: int,
    demo_index: int,
) -> list[str]:
    h5_path = _find_robocasa_h5(data_path, file_index=file_index)
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Expected top-level 'data' group in {h5_path}")
        data_group = f["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if not demo_keys:
            raise ValueError(f"No demos found in {h5_path}")
        if demo_index < 0 or demo_index >= len(demo_keys):
            raise IndexError(f"demo_index={demo_index} out of range [0, {len(demo_keys) - 1}]")
        demo = data_group[demo_keys[demo_index]]
        if "obs" not in demo:
            raise KeyError(f"Expected demo['obs'] in {h5_path}:{demo_keys[demo_index]}")
        return [k for k in demo["obs"].keys() if k.endswith("_image")]


def _default_robocasa_camera_keys(
    data_path: Path,
    file_index: int,
    demo_index: int,
) -> list[str]:
    image_keys = _get_robocasa_image_keys(data_path, file_index=file_index, demo_index=demo_index)
    if not image_keys:
        raise KeyError("No '*_image' datasets found under demo['obs']")

    selected = []

    def _pick_first(candidates: list[str]) -> None:
        for cand in candidates:
            if cand in image_keys and cand not in selected:
                selected.append(cand)
                return

    # Default to two views when possible: gripper (eye-in-hand) + third-person.
    _pick_first(["robot0_eye_in_hand_image", "eye_in_hand_image", "wrist_image"])
    _pick_first(
        [
            "robot0_frontview_image",
            "frontview_image",
            "agentview_image",
            "robot0_robotview_image",
            "robot0_leftview_image",
            "robot0_rightview_image",
        ]
    )

    # Fallback: ensure at least one valid camera key is selected.
    if not selected:
        selected.append(sorted(image_keys)[0])
    return selected


def _load_robocasa_frames(
    data_path: Path,
    file_index: int,
    demo_index: int,
    camera_key: Optional[str],
) -> Tuple[np.ndarray, dict]:
    h5_path = _find_robocasa_h5(data_path, file_index=file_index)
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Expected top-level 'data' group in {h5_path}")
        data_group = f["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if not demo_keys:
            raise ValueError(f"No demos found in {h5_path}")
        if demo_index < 0 or demo_index >= len(demo_keys):
            raise IndexError(f"demo_index={demo_index} out of range [0, {len(demo_keys) - 1}]")
        demo_key = demo_keys[demo_index]
        demo = data_group[demo_key]
        if "obs" not in demo:
            raise KeyError(f"Expected demo['obs'] in {h5_path}:{demo_key}")
        obs = demo["obs"]
        selected_camera = _select_camera_key(obs, camera_key)
        frames = np.array(obs[selected_camera])
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [T,H,W,3], got {frames.shape}")
    return frames, {
        "h5_path": str(h5_path),
        "demo_key": demo_key,
        "camera_key": selected_camera,
    }


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _patch_size_from_model(model: torch.nn.Module) -> int:
    patch = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if patch is None:
        return 16
    if isinstance(patch, (tuple, list)):
        return int(patch[0])
    return int(patch)


def _adjust_size_to_patch(size: int, patch_size: int) -> int:
    return int(math.ceil(size / patch_size) * patch_size)


def _preprocess_video(
    frames_uint8: np.ndarray,
    input_size: int,
    patch_size: int,
    frame_start: int,
    frame_stride: int,
    max_frames: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if frames_uint8.dtype != np.uint8:
        frames_uint8 = np.clip(frames_uint8, 0, 255).astype(np.uint8)

    frames = torch.from_numpy(frames_uint8).float() / 255.0  # [T,H,W,C]
    frames = frames[frame_start::frame_stride]
    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]
    if frames.shape[0] == 0:
        raise ValueError("No frames selected after frame_start/frame_stride/max_frames")

    frames = frames.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
    h, w = frames.shape[-2:]
    side = min(h, w)
    top = int(round((h - side) / 2.0))
    left = int(round((w - side) / 2.0))
    frames = frames[:, :, top : top + side, left : left + side]

    target_size = _adjust_size_to_patch(input_size, patch_size)
    if target_size != side:
        frames = F.interpolate(frames, size=(target_size, target_size), mode="bicubic", align_corners=False)
    vis = frames.clamp(0.0, 1.0)

    norm = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return vis, norm


def _extract_patch_tokens(
    model: torch.nn.Module,
    batch: torch.Tensor,  # [B,C,H,W]
    patch_size: int,
) -> torch.Tensor:
    feats = model.forward_features(batch)

    if isinstance(feats, dict):
        for key in ("x_norm_patchtokens", "x_norm"):
            if key in feats and torch.is_tensor(feats[key]):
                feats = feats[key]
                break
        if isinstance(feats, dict):
            feats = next(v for v in feats.values() if torch.is_tensor(v))
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim != 3:
        raise ValueError(f"Expected [B,N,C] tokens, got {feats.shape}")

    h = batch.shape[-2] // patch_size
    w = batch.shape[-1] // patch_size
    expected = h * w
    prefix = feats.shape[1] - expected
    if prefix < 0:
        raise ValueError(f"Token count {feats.shape[1]} smaller than expected patch count {expected}")
    patch_tokens = feats[:, prefix:, :]
    return patch_tokens.reshape(batch.shape[0], h, w, patch_tokens.shape[-1])


def _fit_pca(samples: torch.Tensor) -> PCAProjector:
    # samples: [N, C]
    if samples.ndim != 2:
        raise ValueError(f"Expected [N,C] samples, got {samples.shape}")
    samples = samples.float()
    mean = samples.mean(dim=0)
    xc = samples - mean

    if xc.shape[1] < 3:
        comp = torch.eye(xc.shape[1], device=xc.device, dtype=xc.dtype)
        pad = torch.zeros((3 - comp.shape[0], comp.shape[1]), device=xc.device, dtype=xc.dtype)
        comp = torch.cat([comp, pad], dim=0)
    else:
        _, _, vh = torch.linalg.svd(xc.cpu(), full_matrices=False)
        comp = vh[:3].to(xc.device)

    proj = xc @ comp.T
    low = torch.quantile(proj, 0.01, dim=0)
    high = torch.quantile(proj, 0.99, dim=0)
    high = torch.maximum(high, low + 1e-6)
    return PCAProjector(mean=mean, components=comp, low=low, high=high)


def _colorize_token_map(token_map: torch.Tensor, projector: PCAProjector, out_hw: Tuple[int, int]) -> np.ndarray:
    # token_map: [h, w, c]
    h, w, c = token_map.shape
    flat = token_map.reshape(-1, c)
    rgb = (flat - projector.mean) @ projector.components.T
    rgb = (rgb - projector.low) / (projector.high - projector.low + 1e-6)
    rgb = rgb.clamp(0.0, 1.0).reshape(h, w, 3)

    rgb = rgb.permute(2, 0, 1).unsqueeze(0)
    rgb = F.interpolate(rgb, size=out_hw, mode="nearest")
    rgb = rgb.squeeze(0).permute(1, 2, 0)
    return rgb.cpu().numpy()


def _collect_pca_samples(
    model: torch.nn.Module,
    video_norm: torch.Tensor,
    patch_size: int,
    batch_size: int,
    max_pca_samples: int,
) -> torch.Tensor:
    sample_chunks = []
    remaining = max_pca_samples
    with torch.no_grad():
        for start in range(0, video_norm.shape[0], batch_size):
            end = min(start + batch_size, video_norm.shape[0])
            tokens = _extract_patch_tokens(model, video_norm[start:end], patch_size=patch_size)
            flat = tokens.reshape(-1, tokens.shape[-1])
            if remaining <= 0:
                continue
            if flat.shape[0] > remaining:
                idx = torch.randperm(flat.shape[0], device=flat.device)[:remaining]
                flat = flat[idx]
            sample_chunks.append(flat.detach().cpu())
            remaining -= flat.shape[0]
            if remaining <= 0:
                break
    if not sample_chunks:
        raise ValueError("No PCA samples collected")
    return torch.cat(sample_chunks, dim=0)


def _render_side_by_side_video(
    model: torch.nn.Module,
    video_vis: torch.Tensor,
    video_norm: torch.Tensor,
    patch_size: int,
    batch_size: int,
    projector: PCAProjector,
    out_path: Path,
    fps: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_hw = (video_vis.shape[-2], video_vis.shape[-1])
    with imageio.get_writer(str(out_path), fps=fps, macro_block_size=1) as writer:
        with torch.no_grad():
            for start in range(0, video_norm.shape[0], batch_size):
                end = min(start + batch_size, video_norm.shape[0])
                tokens = _extract_patch_tokens(model, video_norm[start:end], patch_size=patch_size).cpu()
                vis_batch = video_vis[start:end].cpu()
                for i in range(tokens.shape[0]):
                    rgb = vis_batch[i].permute(1, 2, 0).clamp(0.0, 1.0).numpy()
                    pca_rgb = _colorize_token_map(tokens[i], projector, out_hw=frame_hw)
                    frame = np.concatenate([rgb, pca_rgb], axis=1)
                    writer.append_data((frame * 255.0).astype(np.uint8))


def _run_one_source(
    source_name: str,
    frames_uint8: np.ndarray,
    meta: dict,
    model: torch.nn.Module,
    patch_size: int,
    args,
) -> Path:
    video_vis, video_norm = _preprocess_video(
        frames_uint8=frames_uint8,
        input_size=args.input_size,
        patch_size=patch_size,
        frame_start=args.frame_start,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    video_norm = video_norm.to(args.device)

    samples = _collect_pca_samples(
        model=model,
        video_norm=video_norm,
        patch_size=patch_size,
        batch_size=args.batch_size,
        max_pca_samples=args.max_pca_samples,
    )
    # Keep projector on CPU since token maps are moved to CPU during rendering.
    projector = _fit_pca(samples)

    if source_name == "metaworld":
        tag = f"mw_idx{meta['rollout_idx']}_{_sanitize(meta['task'])}"
    else:
        cam = _sanitize(meta["camera_key"])
        demo = _sanitize(meta["demo_key"])
        tag = f"rc_{demo}_{cam}"
    out_path = args.output_dir / f"{tag}.mp4"

    _render_side_by_side_video(
        model=model,
        video_vis=video_vis,
        video_norm=video_norm,
        patch_size=patch_size,
        batch_size=args.batch_size,
        projector=projector,
        out_path=out_path,
        fps=args.fps,
    )
    return out_path


def _default_metaworld_dir() -> Optional[Path]:
    root = Path(os.environ.get("JEPAWM_DSET", ""))
    if not str(root):
        return None
    path = root / "Metaworld" / "data"
    return path if path.exists() else None


def _default_robocasa_path() -> Optional[Path]:
    root = Path(os.environ.get("JEPAWM_DSET", ""))
    if not str(root):
        return None
    file_path = root / "robocasa" / "combine_all_im256.hdf5"
    if file_path.exists():
        return file_path
    dir_path = root / "robocasa"
    return dir_path if dir_path.exists() else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render 1x2 videos: original frame + DINOv3 PCA for Metaworld and/or RoboCasa."
    )
    parser.add_argument("--env", choices=["metaworld", "robocasa", "both"], default="both")

    parser.add_argument("--metaworld-data-dir", type=Path, default=_default_metaworld_dir())
    parser.add_argument("--metaworld-index", type=int, default=0)

    parser.add_argument("--robocasa-path", type=Path, default=_default_robocasa_path())
    parser.add_argument("--robocasa-file-index", type=int, default=0)
    parser.add_argument("--robocasa-demo-index", type=int, default=0)
    parser.add_argument(
        "--robocasa-camera-key",
        type=str,
        default=None,
        help="Specific RoboCasa camera key to render (default: render gripper + third-person when available).",
    )

    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to render. Omit to render the full demo.",
    )

    parser.add_argument("--dino-model", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument("--input-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pca-samples", type=int, default=50000)

    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov3_pca"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be >= 1")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be >= 1 when provided")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.max_pca_samples <= 0:
        raise ValueError("--max-pca-samples must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = _resolve_device(args.device)

    model = timm.create_model(args.dino_model, pretrained=True, dynamic_img_size=True).eval().to(args.device)
    patch_size = _patch_size_from_model(model)

    output_paths = []
    if args.env in ("metaworld", "both"):
        if args.metaworld_data_dir is None:
            raise ValueError("--metaworld-data-dir is required for env=metaworld/both")
        frames, meta = _load_metaworld_frames(args.metaworld_data_dir, args.metaworld_index)
        out = _run_one_source("metaworld", frames, meta, model, patch_size, args)
        print(f"[metaworld] saved: {out}")
        output_paths.append(out)

    if args.env in ("robocasa", "both"):
        if args.robocasa_path is None:
            raise ValueError("--robocasa-path is required for env=robocasa/both")
        if args.robocasa_camera_key:
            camera_keys = [args.robocasa_camera_key]
        else:
            camera_keys = _default_robocasa_camera_keys(
                data_path=args.robocasa_path,
                file_index=args.robocasa_file_index,
                demo_index=args.robocasa_demo_index,
            )

        for camera_key in camera_keys:
            frames, meta = _load_robocasa_frames(
                data_path=args.robocasa_path,
                file_index=args.robocasa_file_index,
                demo_index=args.robocasa_demo_index,
                camera_key=camera_key,
            )
            out = _run_one_source("robocasa", frames, meta, model, patch_size, args)
            print(f"[robocasa:{meta['camera_key']}] saved: {out}")
            output_paths.append(out)

    if output_paths:
        print("\nGenerated files:")
        for path in output_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
