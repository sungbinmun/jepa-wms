import argparse
import math
import os
import pathlib
import sys
from typing import Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import timm
from omegaconf import OmegaConf
from torchvision import transforms as tvt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slotcontrast.data.libero_h5_datamodule import LiberoH5Dataset
from slotcontrast.data.transforms import build_inference_transform
from slotcontrast.visualizations import create_grid_frame_rgb, pca_proj, colorize_map


def _ensure_video_vis_layout(video_vis: torch.Tensor) -> torch.Tensor:
    if video_vis.ndim != 5:
        raise ValueError(f"Expected video_visualization with 5 dims, got {video_vis.shape}")
    if video_vis.shape[1] in (1, 3) and video_vis.shape[2] > 3:
        return video_vis.permute(0, 2, 1, 3, 4)
    return video_vis


def _patch_size_from_model(model: torch.nn.Module) -> int:
    ps = getattr(model, "patch_embed", None)
    if ps is None:
        return 16
    patch = getattr(ps, "patch_size", None)
    if patch is None:
        return 16
    if isinstance(patch, (tuple, list)):
        return patch[0]
    return int(patch)


def _adjust_size_to_patch(size: int, patch_size: int) -> int:
    return int(math.ceil(size / patch_size) * patch_size)


def _center_crop_video(tensor: torch.Tensor) -> torch.Tensor:
    h, w = tensor.shape[-2:]
    side = min(h, w)
    top = int(round((h - side) / 2.0))
    left = int(round((w - side) / 2.0))
    return tensor[:, top : top + side, left : left + side]


def _resize_segmentations(
    segmentations: torch.Tensor, input_size: int, resize_mode: str
) -> torch.Tensor:
    segmentations = _center_crop_video(segmentations)
    segmentations = segmentations.unsqueeze(1).float()
    if resize_mode in ("bilinear", "bicubic", "trilinear", "linear"):
        segmentations = F.interpolate(
            segmentations, size=(input_size, input_size), mode=resize_mode, align_corners=False
        )
    else:
        segmentations = F.interpolate(
            segmentations, size=(input_size, input_size), mode=resize_mode
        )
    segmentations = torch.round(segmentations.squeeze(1)).to(torch.int64)
    return segmentations


def _normalize_id_list(value, default):
    if value is None:
        return list(default) if default is not None else []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def _select_libero_index(dataset: LiberoH5Dataset, input_config) -> int:
    sample_index = input_config.get("sample_index")
    if sample_index is not None:
        return int(sample_index)

    demo = input_config.get("demo")
    file_index = input_config.get("file_index")
    chunk_index = input_config.get("chunk_index")

    if demo is None and file_index is None and chunk_index is None:
        return 0

    requested_view = input_config.get("view_key")
    if requested_view is None:
        rgb_key = input_config.get("rgb_key")
        if isinstance(rgb_key, str):
            requested_view = rgb_key

    for idx, entry in enumerate(dataset.index):
        if len(entry) == 3:
            file_idx, demo_name, ch_idx = entry
            view_key = None
        else:
            file_idx, demo_name, ch_idx, view_key = entry
        if demo is not None and demo_name != demo:
            continue
        if file_index is not None and file_idx != file_index:
            continue
        if chunk_index is not None and ch_idx != chunk_index:
            continue
        if requested_view is not None and view_key is not None and view_key != requested_view:
            continue
        return idx

    raise ValueError(
        "No matching Libero sample found. "
        "Check demo/file_index/chunk_index or use sample_index."
    )


def prepare_libero_h5(input_config, include_segmentations: bool):
    if input_config.get("h5_path") is None:
        raise ValueError("Libero input requires `input.h5_path` in the config.")
    if input_config.get("transforms") is None:
        raise ValueError("Libero input requires `input.transforms` to set input_size.")

    transforms_cfg = input_config.transforms
    if not OmegaConf.is_config(transforms_cfg):
        transforms_cfg = OmegaConf.create(transforms_cfg)

    dataset = LiberoH5Dataset(
        h5_path=input_config.h5_path,
        transforms_dict=None,
        rgb_key=input_config.get("rgb_key"),
        rgb_key_sampling=input_config.get("rgb_key_sampling", "first"),
        segmentation_key=input_config.get("segmentation_key", "segmentation"),
        include_segmentations=include_segmentations,
        include_actions=False,
        h5_prefixes=input_config.get("h5_prefixes"),
        use_chunks=input_config.get("use_chunks", False),
        chunk_size=input_config.get("chunk_size", 4),
        sample_one_chunk_per_video=input_config.get("sample_one_chunk_per_video", False),
        temporal_stride=input_config.get("temporal_stride", 1),
    )

    sample_idx = _select_libero_index(dataset, input_config)
    sample = dataset[sample_idx]
    video = sample["video"]
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    if video.dtype == torch.uint8:
        video = video.float() / 255.0

    frame_start = int(input_config.get("frame_start", 0))
    frame_end = input_config.get("frame_end")
    if frame_end is not None:
        frame_end = int(frame_end)
    max_frames = input_config.get("max_frames")
    if max_frames is not None:
        max_frames = int(max_frames)
        max_end = frame_start + max_frames
        frame_end = max_end if frame_end is None else min(frame_end, max_end)

    if frame_start or frame_end is not None:
        video = video[frame_start:frame_end]

    input_size = int(transforms_cfg.get("input_size"))
    video_vis = video.permute(0, 3, 1, 2)
    video_vis = tvt.Resize((input_size, input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)

    tfs = build_inference_transform(transforms_cfg)
    video = video.permute(3, 0, 1, 2)
    video = tfs(video).permute(1, 0, 2, 3)

    inputs = {"video": video.unsqueeze(0), "video_visualization": video_vis.unsqueeze(0)}
    if include_segmentations:
        segmentations = sample.get("segmentations")
        if segmentations is None:
            raise KeyError("Segmentations requested but not found in the HDF5 sample.")
        if isinstance(segmentations, np.ndarray):
            segmentations = torch.from_numpy(segmentations)
        if segmentations.ndim == 4 and segmentations.shape[-1] == 1:
            segmentations = segmentations[..., 0]
        if segmentations.dtype == torch.uint8:
            segmentations = segmentations.float()
        if frame_start or frame_end is not None:
            segmentations = segmentations[frame_start:frame_end]
        seg_resize_mode = input_config.get("segmentation_resize_mode", "nearest")
        segmentations = _resize_segmentations(segmentations, input_size, seg_resize_mode)
        inputs["segmentations"] = segmentations.unsqueeze(0)
    return inputs


def _forward_tokens(
    model: torch.nn.Module, x: torch.Tensor, patch_size: int
) -> Tuple[torch.Tensor, int, int]:
    H, W = x.shape[-2], x.shape[-1]
    h = H // patch_size
    w = W // patch_size
    p_expected = h * w

    feats = model.forward_features(x)
    if isinstance(feats, dict):
        for key in ("x_norm_patchtokens", "x_norm"):
            if key in feats and torch.is_tensor(feats[key]):
                feats = feats[key]
                break
        if isinstance(feats, dict):
            feats = next(v for v in feats.values() if torch.is_tensor(v))
    if isinstance(feats, (tuple, list)):
        feats = feats[0]

    bsz, p_total, dim = feats.shape
    prefix = p_total - p_expected
    if prefix < 0:
        raise ValueError(f"Expected at least {p_expected} tokens, got {p_total}.")
    feats = feats[:, prefix:]
    return feats.reshape(bsz, h, w, dim), h, w


def _resolve_device(config, cli_gpu):
    if cli_gpu is not None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cli_gpu}")
        return torch.device("cpu")
    device = config.get("device")
    if device:
        return torch.device(device)
    gpu_index = config.get("gpu_index")
    if gpu_index is not None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(gpu_index)}")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_input_config(input_config):
    if input_config.get("class") and not input_config.get("h5_prefixes"):
        input_config.h5_prefixes = input_config.get("class")
    if input_config.get("view_key") and not input_config.get("rgb_key"):
        input_config.rgb_key = input_config.get("view_key")


def _resize_mask_to_tokens(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=(h, w), mode="nearest")
    return mask.squeeze(0).squeeze(0) > 0.5


def _build_foreground_mask(segmentation: torch.Tensor, ignore_ids) -> torch.Tensor:
    fg_mask = torch.ones_like(segmentation, dtype=torch.bool)
    for ignore_id in ignore_ids:
        fg_mask &= segmentation != ignore_id
    return fg_mask


def main(config, cli_gpu=None):
    if config.get("input") is None:
        raise ValueError("Missing `input` section in config.")
    if config.get("output") is None:
        raise ValueError("Missing `output` section in config.")

    _normalize_input_config(config.input)

    device = _resolve_device(config, cli_gpu)
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("dino_model", "vit_base_patch16_dinov3")
    model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True).eval().to(device)
    patch_size = _patch_size_from_model(model)

    transforms_cfg = config.input.transforms
    if not OmegaConf.is_config(transforms_cfg):
        transforms_cfg = OmegaConf.create(transforms_cfg)
        config.input.transforms = transforms_cfg

    input_size = int(transforms_cfg.get("input_size", 224))
    target_size = _adjust_size_to_patch(input_size, patch_size)
    if target_size != input_size:
        print(
            f"Adjusting input_size from {input_size} to {target_size} to fit patch size {patch_size}."
        )
        transforms_cfg.input_size = target_size

    use_fg_pca = bool(config.output.get("enable_fg_pca", False))
    inputs = prepare_libero_h5(config.input, include_segmentations=use_fg_pca)
    video = inputs["video"]
    video_vis = _ensure_video_vis_layout(inputs["video_visualization"])
    segmentations = inputs.get("segmentations") if use_fg_pca else None

    frame_size = tuple(video_vis.shape[-2:])
    num_frames = min(video.shape[1], video_vis.shape[1])
    if segmentations is not None:
        num_frames = min(num_frames, segmentations.shape[1])
    batch_size = int(model_cfg.get("batch_size", 8))
    fps = int(config.output.get("fps", 10))
    padding = int(config.output.get("padding", 2))

    output_path = config.output.get("path") or config.output.get("save_path")
    if not output_path:
        raise ValueError("Output requires `output.path` or `output.save_path`.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    frames = []
    proj = None
    proj_fg = None
    ignore_ids = set(_normalize_id_list(config.input.get("bg_ids"), [0, 1]))
    ignore_ids.update(_normalize_id_list(config.input.get("agent_ids"), [65535]))
    ignore_ids.update(_normalize_id_list(config.input.get("ignore_ids"), []))
    with torch.no_grad():
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            batch = video[0, start:end].to(device)
            feats_map, h, w = _forward_tokens(model, batch, patch_size)
            feats_map = feats_map.cpu()
            if proj is None:
                proj = pca_proj(feats_map[0].reshape(-1, feats_map.shape[-1]))
            if use_fg_pca and proj_fg is None:
                for i in range(feats_map.shape[0]):
                    seg_frame = segmentations[0, start + i]
                    fg_mask_full = _build_foreground_mask(seg_frame, ignore_ids)
                    fg_mask_tokens = _resize_mask_to_tokens(fg_mask_full, h, w)
                    if fg_mask_tokens.any():
                        fg_feats = feats_map[i][fg_mask_tokens]
                        proj_fg = pca_proj(fg_feats.reshape(-1, feats_map.shape[-1]))
                        break
                if proj_fg is None:
                    proj_fg = proj
            for i in range(feats_map.shape[0]):
                frame = video_vis[0, start + i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                dino_rgb = colorize_map(feats_map[i], proj, out_hw=frame_size)
                grid_frames = [frame, dino_rgb]
                if use_fg_pca:
                    seg_frame = segmentations[0, start + i]
                    fg_mask_full = _build_foreground_mask(seg_frame, ignore_ids)
                    dino_fg_rgb = colorize_map(feats_map[i], proj_fg, out_hw=frame_size)
                    fg_mask_np = fg_mask_full.cpu().numpy().astype(np.float32)
                    dino_fg_rgb = dino_fg_rgb * fg_mask_np[..., None]
                    grid_frames.append(dino_fg_rgb)
                grid_cols = len(grid_frames)
                grid = create_grid_frame_rgb(
                    grid_frames,
                    grid_size=(1, grid_cols),
                    image_size=frame_size,
                    padding=padding,
                )
                frames.append((grid * 255).astype(np.uint8))

    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved Libero DINOv3 PCA video to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a Libero HDF5 demo as input + DINOv3 PCA feature video."
    )
    parser.add_argument(
        "--config",
        default="configs/inference/libero_dinov3_pca.yml",
        help="Path to YAML config.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index (overrides config).")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg, cli_gpu=args.gpu)
