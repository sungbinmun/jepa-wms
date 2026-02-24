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


def compute_proj(feats: torch.Tensor) -> torch.Tensor:
    """Compute a fixed 3xD projection (PCA) for colorization."""
    flat = feats.reshape(-1, feats.shape[-1])
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


def load_models(model_v2: str, model_v3: str, device: torch.device):
    kwargs = dict(pretrained=True, dynamic_img_size=True)
    m2 = timm.create_model(model_v2, **kwargs).eval().to(device)
    m3 = timm.create_model(model_v3, **kwargs).eval().to(device)
    ps2 = _patch_size_from_model(m2)
    ps3 = _patch_size_from_model(m3)
    return m2, m3, ps2, ps3


def forward_tokens(model: torch.nn.Module, x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
    """Return patch tokens without prefixes (CLS/register) and spatial dims based on input size and patch."""
    H, W = x.shape[-2], x.shape[-1]
    h = H // patch_size
    w = W // patch_size
    P_expected = h * w

    feats = model.forward_features(x)  # (B, tokens, D)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    B, P_total, D = feats.shape
    prefix = P_total - P_expected
    if prefix < 0:
        raise ValueError(f"Expected at least {P_expected} tokens, got {P_total}.")
    feats = feats[:, prefix:]  # drop cls/register tokens
    return feats.reshape(B, h, w, D), h, w


def prepare_inputs(path: str, input_type: str, input_size: int, use_movi_norm: bool):
    if input_type == "video":
        video, _, _ = read_video(path)
        video = video.float() / 255.0  # T, H, W, C
        vis_resize = CropResize(
            dataset_type="video", crop_type="central", size=input_size, resize_mode="bilinear"
        )
        video_vis = vis_resize(video.permute(3, 0, 1, 2)).permute(1, 0, 2, 3)  # F, C, H, W

        tf_cfg = {"input_size": input_size, "dataset_type": "video", "use_movi_normalization": use_movi_norm}
        tfs = build_inference_transform(OmegaConf.create(tf_cfg))
        video_feat = tfs(video.permute(3, 0, 1, 2)).permute(1, 0, 2, 3)  # F, C, H, W
        video_vis = video_vis.unsqueeze(0)  # 1, T, C, H, W
        video_feat = video_feat.unsqueeze(0)
        return video_vis, video_feat
    else:
        img = read_image(path).float() / 255.0
        vis_resize = CropResize(
            dataset_type="image", crop_type="short_side_resize_central", size=input_size, resize_mode="bilinear"
        )
        img_vis = vis_resize(img)
        tf_cfg = {"input_size": input_size, "dataset_type": "image", "use_movi_normalization": use_movi_norm}
        tfs = build_inference_transform(OmegaConf.create(tf_cfg))
        img_feat = tfs(img)
        img_vis = img_vis.unsqueeze(0).unsqueeze(1)  # 1,1,C,H,W
        img_feat = img_feat.unsqueeze(0).unsqueeze(1)
        return img_vis, img_feat


def _adjust_size_to_patch(size: int, patch_sizes):
    lcm = 1
    for p in patch_sizes:
        lcm = lcm * p // math.gcd(lcm, p)
    return int(math.ceil(size / lcm) * lcm)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    m2, m3, ps2, ps3 = load_models(args.dinov2_model, args.dinov3_model, device)
    target_size = _adjust_size_to_patch(args.input_size, [ps2, ps3])
    if target_size != args.input_size:
        print(f"Adjusting input_size from {args.input_size} to {target_size} to fit patch sizes ({ps2}, {ps3}).")

    video_vis, video_feat = prepare_inputs(args.input, args.type, target_size, args.use_movi_norm)
    B, T, C, H, W = video_vis.shape
    assert B == 1, "Batch size must be 1 for visualization."

    frames = []
    proj2 = None
    proj3 = None
    for t in range(T):
        x = video_feat[:, t].to(device)
        with torch.no_grad():
            f2, h2, w2 = forward_tokens(m2, x, ps2)
            f3, h3, w3 = forward_tokens(m3, x, ps3)
        f2 = f2.cpu()
        f3 = f3.cpu()
        if proj2 is None:
            proj2 = compute_proj(f2[0])
        if proj3 is None:
            proj3 = compute_proj(f3[0])
        v = video_vis[0, t].permute(1, 2, 0).cpu().numpy()
        v2_rgb = colorize_feature_map(f2[0], h2, w2, H, W, proj2)
        v3_rgb = colorize_feature_map(f3[0], h3, w3, H, W, proj3)
        grid = create_grid_frame_rgb([v, v2_rgb, v3_rgb], grid_size=(1, 3), image_size=(H, W))
        frames.append((grid * 255).astype(np.uint8))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved comparison video to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DINOv2 vs DINOv3 feature maps on the same input.")
    parser.add_argument("--input", required=True, help="Path to video or image.")
    parser.add_argument("--type", choices=["video", "image"], default="video")
    parser.add_argument("--dinov2-model", default="vit_base_patch14_dinov2")
    parser.add_argument("--dinov3-model", default="vit_base_patch16_dinov3")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--use-movi-norm", action="store_true", help="Use MOVi normalization instead of ImageNet.")
    parser.add_argument("--output", default="outputs/dino_v2_v3.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", help="cpu or cuda (default: auto)")
    args = parser.parse_args()
    main(args)
