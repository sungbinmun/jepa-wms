import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional
import math

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from slotcontrast import configuration, models
from slotcontrast.data import datamodules
from slotcontrast.data.transforms import Denormalize
from slotcontrast.visualizations import color_map, draw_segmentation_masks_on_image


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Rollout visualization for SlotFormer dynamics on DROID."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config (e.g., configs/slotcontrast/droid_mix_pred.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to override config.model.load_weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/slotformer_rollout.mp4"),
        help="Where to save the rollout video.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root. Defaults to SLOTCONTRAST_DATA_PATH or ./data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device, defaults to cuda if available else cpu.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override dataset.train_pipeline.chunk_size (controls sequence length).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Trim the sampled clip to at most this many frames.",
    )
    parser.add_argument(
        "--slot-indices",
        type=int,
        nargs="*",
        default=[],
        help="Slot indices to overlay after the 1x3 grid (e.g., --slot-indices 0 3).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for the saved video.",
    )
    return parser.parse_args()


def _override_config_path(config_path: Path, args) -> OmegaConf:
    overrides: List[str] = []
    # Force single-sample batch for deterministic visualization.
    overrides.append("dataset.batch_size=1")
    overrides.append("dataset.num_workers=0")
    overrides.append("dataset.num_val_workers=0")
    if args.chunk_size:
        overrides.append(f"dataset.train_pipeline.chunk_size={args.chunk_size}")
    if args.checkpoint:
        overrides.append(f"model.load_weights={args.checkpoint}")
    return configuration.load_config(config_path, overrides=overrides)


def _select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tokens_from_feats(feats: torch.Tensor) -> torch.Tensor:
    """Return tokens as [T, P, D] from either [T, P, D] or [T, D, H, W]."""
    if feats.ndim == 3:
        return feats
    if feats.ndim == 4:
        feats = feats.permute(0, 2, 3, 1)  # T, H, W, D
        t, h, w, d = feats.shape
        return feats.reshape(t, h * w, d)
    raise ValueError(f"Unexpected feature shape {feats.shape}")


def _pca_project_to_rgb(
    feats_tokens: torch.Tensor,
    basis_tokens: torch.Tensor,
    target_hw: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """
    Project tokens to 3-channel PCA image using basis from basis_tokens.
    feats_tokens: [T, P, D], basis_tokens: [T_ref, P, D]
    Returns: [T, 3, H, W] float in [0,1]
    """
    t_ref, p_ref, d = basis_tokens.shape
    h = int(math.sqrt(p_ref))
    w = h
    if h * w != p_ref:
        raise ValueError(f"Patch count {p_ref} is not a square number.")

    # Fit PCA on basis_tokens (flatten over time and patches)
    basis_flat = basis_tokens.reshape(t_ref * p_ref, d)
    basis_mean = basis_flat.mean(0, keepdim=True)
    basis_centered = basis_flat - basis_mean
    _, _, v = torch.pca_lowrank(basis_centered, q=min(3, d))
    comps = v[:, :3]  # D x 3

    # Project feats
    t, p, d_feats = feats_tokens.shape
    if p != p_ref:
        raise ValueError(f"Token count mismatch: {p} vs {p_ref}")
    feats_flat = feats_tokens.reshape(t * p, d_feats)
    feats_centered = feats_flat - basis_mean
    proj = feats_centered @ comps  # (t*p, 3)

    proj = proj.reshape(t, h, w, 3).permute(0, 3, 1, 2)  # T,3,H,W

    # Normalize to [0,1] over all frames/channels
    proj_min = proj.amin(dim=(0, 2, 3), keepdim=True)
    proj_max = proj.amax(dim=(0, 2, 3), keepdim=True)
    proj = (proj - proj_min) / (proj_max - proj_min + 1e-6)

    if target_hw is not None and (target_hw[0] != h or target_hw[1] != w):
        proj = F.interpolate(proj, size=target_hw, mode="bilinear", align_corners=False)

    return proj.clamp(0, 1)


def _overlay_single_slot(video: torch.Tensor, mask: torch.Tensor, color: tuple) -> torch.Tensor:
    """
    Overlay a single slot mask on a video or a single frame.
    video: [T, 3, H, W] or [3, H, W] float in [0,1]
    mask:  [T, H, W] or [H, W] bool/float
    color: (R, G, B) in [0, 255]
    """
    single_frame = video.ndim == 3
    if single_frame:
        video = video.unsqueeze(0)
        mask = mask.unsqueeze(0)

    video_uint8 = (video.clamp(0, 1) * 255).to(torch.uint8)
    if mask.dtype != torch.bool:
        mask = mask > 0.5

    frames = []
    for frame, m in zip(video_uint8, mask):
        frames.append(
            draw_segmentation_masks_on_image(
                frame, m, colors=[color], alpha=0.5
            )
        )
    stacked = torch.stack(frames)
    if single_frame:
        return stacked[0]
    return stacked


def _to_uint8(frame: torch.Tensor) -> np.ndarray:
    """frame: [3, H, W] float or uint8 -> np.uint8 HWC."""
    if frame.dtype != torch.uint8:
        frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
    return frame.permute(1, 2, 0).cpu().numpy()


def main():
    args = _parse_args()
    cfg = _override_config_path(args.config, args)
    device = _select_device(args.device)
    data_root = args.data_dir or os.environ.get("SLOTCONTRAST_DATA_PATH", "data")

    model = models.build(cfg.model, cfg.optimizer)
    model.eval()
    model.to(device)

    dm = datamodules.build(cfg.dataset, data_dir=str(data_root))
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    # Trim frames if requested.
    if args.max_frames:
        batch["video"] = batch["video"][:, : args.max_frames]
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(batch)
        aux_outputs = model.aux_forward(batch, outputs)

    history_len = getattr(model.dynamics_predictor, "history_len", 0)
    denorm = Denormalize("video")
    gt_video = denorm(batch["video"].detach().cpu()[0]).clamp(0, 1)  # [T, C, H, W]

    # GT DINO features (tokens)
    enc_feats = outputs["encoder"]["backbone_features"][0].detach().cpu()  # [T, P, D] or [T,D,H,W]
    gt_tokens = _tokens_from_feats(enc_feats)

    # Decoder recon tokens (teacher-forced)
    recon_tokens = _tokens_from_feats(outputs["decoder"]["reconstruction"][0].detach().cpu())

    # Predicted recon tokens (rollout)
    pred_tokens = outputs["decoder"].get("predicted_reconstruction")
    if pred_tokens is None:
        raise RuntimeError("predicted_reconstruction not found. Make sure dynamics_predictor is set.")
    pred_tokens = _tokens_from_feats(pred_tokens[0].detach().cpu())

    t_total = recon_tokens.shape[0]
    rollout = pred_tokens.shape[0]
    start_pred = max(t_total - rollout, history_len)

    # Align predicted tokens to full length: use teacher-forced before start_pred, predicted after
    pred_full_tokens = recon_tokens.clone()
    aligned_len_pred = min(rollout, t_total - start_pred)
    pred_full_tokens[start_pred : start_pred + aligned_len_pred] = pred_tokens[:aligned_len_pred]

    h_vis, w_vis = gt_video.shape[-2:]

    # PCA basis from GT tokens; apply to both GT and predicted tokens.
    gt_pca = _pca_project_to_rgb(gt_tokens, basis_tokens=gt_tokens, target_hw=(h_vis, w_vis))
    pred_pca = _pca_project_to_rgb(pred_full_tokens, basis_tokens=gt_tokens, target_hw=(h_vis, w_vis))

    # Prepare masks for optional slot overlays.
    slot_masks = aux_outputs.get("dynamics_predictor_masks_vis_hard")
    mask_full = None
    if slot_masks is not None:
        slot_masks = slot_masks[0].detach().cpu()  # [rollout, S, H, W] expected
        # Pad to full length with zeros before prediction start.
        mask_full = torch.zeros(
            (t_total, slot_masks.shape[1], slot_masks.shape[2], slot_masks.shape[3]),
            dtype=slot_masks.dtype,
        )
        aligned_len_mask = min(slot_masks.shape[0], t_total - start_pred)
        mask_full[start_pred : start_pred + aligned_len_mask] = slot_masks[:aligned_len_mask]
        # Resize to match video resolution if needed.
        if mask_full.shape[-2:] != (h_vis, w_vis):
            mask_full = F.interpolate(mask_full, size=(h_vis, w_vis), mode="nearest")

    cmap = color_map(mask_full.shape[1] if mask_full is not None else 1)
    frames = []
    for t in range(t_total):
        row_imgs: List[np.ndarray] = []
        row_imgs.append(_to_uint8(gt_video[t]))
        row_imgs.append(_to_uint8(gt_pca[t]))
        row_imgs.append(_to_uint8(pred_pca[t]))

        if mask_full is not None and args.slot_indices:
            for idx in args.slot_indices:
                if idx >= mask_full.shape[1]:
                    raise ValueError(
                        f"Requested slot index {idx} but only {mask_full.shape[1]} slots."
                    )
                overlay = _overlay_single_slot(
                    gt_video[t],
                    mask_full[t, idx],
                    color=tuple(int(c) for c in cmap[idx]),
                )
                row_imgs.append(_to_uint8(overlay))

        frames.append(np.concatenate(row_imgs, axis=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved rollout video to {args.output}")


if __name__ == "__main__":
    main()
