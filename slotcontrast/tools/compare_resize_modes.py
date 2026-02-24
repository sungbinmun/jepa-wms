import argparse
import math
import os
from typing import Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import timm
from omegaconf import OmegaConf
from torchvision.io import read_image, read_video

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slotcontrast.data.transforms import CropResize, build_inference_transform
from slotcontrast.visualizations import create_grid_frame_rgb


def compute_proj_from_flat(flat: torch.Tensor) -> torch.Tensor:
    """Compute a fixed 3xD projection (PCA) for colorization."""
    flat = flat - flat.mean(dim=0, keepdim=True)
    if flat.shape[1] >= 3:
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        proj = vh[:3]
    else:
        proj = torch.eye(flat.shape[1], device=flat.device)
    return proj


def colorize_feature_map(
    feats: torch.Tensor, h: int, w: int, H: int, W: int, proj: torch.Tensor
) -> np.ndarray:
    """Project spatial feature map (h, w, d) to RGB via provided projection and upsample to (H, W)."""
    flat = feats.reshape(h * w, -1)
    flat = flat - flat.mean(dim=0, keepdim=True)
    proj = proj.to(flat.device)
    rgb = flat @ proj.T  # (P, 3)
    rgb = rgb.reshape(h, w, -1)
    if rgb.shape[2] < 3:
        pad = torch.zeros((h, w, 3 - rgb.shape[2]), device=rgb.device)
        rgb = torch.cat([rgb, pad], dim=2)
    rgb_min = rgb.amin(dim=(0, 1), keepdim=True)
    rgb_max = rgb.amax(dim=(0, 1), keepdim=True)
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # 1,3,h,w
    rgb = F.interpolate(rgb, size=(H, W), mode="nearest")
    return rgb.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


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


def forward_tokens(model: torch.nn.Module, x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
    """Return patch tokens without prefixes (CLS/register) and spatial dims based on input size and patch."""
    H, W = x.shape[-2], x.shape[-1]
    h = H // patch_size
    w = W // patch_size
    P_expected = h * w

    feats = model.forward_features(x)  # (B, tokens, D) or dict
    if isinstance(feats, dict):
        for key in ("x_norm_patchtokens", "x_norm"):
            if key in feats and torch.is_tensor(feats[key]):
                feats = feats[key]
                break
        if isinstance(feats, dict):
            feats = next(v for v in feats.values() if torch.is_tensor(v))
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    B, P_total, D = feats.shape
    prefix = P_total - P_expected
    if prefix < 0:
        raise ValueError(f"Expected at least {P_expected} tokens, got {P_total}.")
    feats = feats[:, prefix:]  # drop cls/register tokens
    return feats.reshape(B, h, w, D), h, w


def _build_transform(dataset_type: str, input_size: int, resize_mode: str, use_movi_norm: bool):
    cfg = OmegaConf.create(
        {
            "input_size": input_size,
            "dataset_type": dataset_type,
            "resize_mode": resize_mode,
            "clamp_zero_one": True,
            "use_movi_normalization": use_movi_norm,
        }
    )
    return build_inference_transform(cfg)


def _adjust_size_to_patch(size: int, patch_size: int):
    return int(math.ceil(size / patch_size) * patch_size)


def prepare_inputs(path: str, input_type: str, input_size: int, use_movi_norm: bool):
    dataset_type = "video" if input_type == "video" else "image"
    vis_resize = CropResize(
        dataset_type=dataset_type,
        crop_type="central",
        size=input_size,
        resize_mode="bilinear",
        clamp_zero_one=True,
    )
    tf_bilinear = _build_transform(dataset_type, input_size, "bilinear", use_movi_norm)
    tf_bicubic = _build_transform(dataset_type, input_size, "bicubic", use_movi_norm)

    if input_type == "video":
        video, _, _ = read_video(path)
        video = video.float() / 255.0  # T, H, W, C
        video_cfhw = video.permute(3, 0, 1, 2)  # C, F, H, W

        vis = vis_resize(video_cfhw).permute(1, 0, 2, 3)  # F, C, H, W
        bilinear = tf_bilinear(video_cfhw).permute(1, 0, 2, 3)
        bicubic = tf_bicubic(video_cfhw).permute(1, 0, 2, 3)
    else:
        image = read_image(path).float() / 255.0  # C, H, W
        vis = vis_resize(image).unsqueeze(0)  # 1, C, H, W
        bilinear = tf_bilinear(image).unsqueeze(0)
        bicubic = tf_bicubic(image).unsqueeze(0)

    return vis, bilinear, bicubic


def save_frames(frames, output: str, fps: int):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    root, ext = os.path.splitext(output)
    ext = ext.lower()

    # Default to a sensible extension if none was provided
    if ext == "":
        ext = ".mp4" if len(frames) > 1 else ".png"
        output = root + ext
        print(f"No extension provided; defaulting to {output}")

    if ext in (".png", ".jpg", ".jpeg"):
        imageio.imwrite(output, frames[0])
        print(f"Saved image to {output}")
    else:
        with imageio.get_writer(output, fps=fps, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Saved video to {output}")


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = timm.create_model(args.model, pretrained=True, dynamic_img_size=True).eval().to(device)
    patch_size = _patch_size_from_model(model)

    target_size = _adjust_size_to_patch(args.input_size, patch_size)
    if target_size != args.input_size:
        print(f"Adjusting input_size from {args.input_size} to {target_size} to fit patch size {patch_size}.")

    vis, bilinear, bicubic = prepare_inputs(args.input, args.type, target_size, args.use_movi_norm)
    T = vis.shape[0]
    H, W = vis.shape[-2], vis.shape[-1]
    assert (
        bilinear.shape == bicubic.shape == vis.shape
    ), f"Input shapes must match: vis={vis.shape}, bilinear={bilinear.shape}, bicubic={bicubic.shape}"

    frames = []
    proj = None
    for t in range(T):
        x_bilinear = bilinear[t].unsqueeze(0).to(device)
        x_bicubic = bicubic[t].unsqueeze(0).to(device)
        with torch.no_grad():
            f_bilinear, h1, w1 = forward_tokens(model, x_bilinear, patch_size)
            f_bicubic, h2, w2 = forward_tokens(model, x_bicubic, patch_size)
        if proj is None:
            flat = torch.cat(
                [f_bilinear[0].reshape(-1, f_bilinear.shape[-1]), f_bicubic[0].reshape(-1, f_bicubic.shape[-1])],
                dim=0,
            )
            proj = compute_proj_from_flat(flat)

        v = vis[t].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        bil_rgb = colorize_feature_map(f_bilinear[0], h1, w1, H, W, proj)
        bic_rgb = colorize_feature_map(f_bicubic[0], h2, w2, H, W, proj)

        grid = create_grid_frame_rgb([v, bil_rgb, bic_rgb], grid_size=(1, 3), image_size=(H, W))
        frames.append((grid * 255).astype(np.uint8))

    save_frames(frames, args.output, args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize DINO features with bilinear vs bicubic resize next to the original frame."
    )
    parser.add_argument("--input", required=True, help="Path to an image or video.")
    parser.add_argument("--type", choices=["image", "video"], default="image")
    parser.add_argument("--model", default="vit_base_patch16_dinov3")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--use-movi-norm", action="store_true", help="Use MOVi normalization instead of ImageNet.")
    parser.add_argument("--output", default="outputs/dino_resize_compare.mp4")
    parser.add_argument("--fps", type=int, default=10, help="FPS for video output.")
    parser.add_argument("--device", help="cpu or cuda (default: auto)")
    main(parser.parse_args())
