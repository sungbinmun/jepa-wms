import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import write_video

from slotcontrast.modules.encoders import TimmExtractor


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pca_proj(x: torch.Tensor, max_samples: int = 20000) -> torch.Tensor:
    if x.numel() == 0:
        return torch.eye(1, device=x.device)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_samples]
        x = x[idx]
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[1] < 3:
        return torch.eye(x.shape[1], device=x.device)
    x_cpu = x.detach().float().cpu()
    _, _, vh = torch.linalg.svd(x_cpu, full_matrices=False)
    return vh[:3].to(x.device)


def colorize_map(x: torch.Tensor, proj: torch.Tensor, out_hw=None) -> np.ndarray:
    h, w, c = x.shape
    flat = x.reshape(-1, c)
    flat = flat - flat.mean(dim=0, keepdim=True)
    if proj.shape[1] != flat.shape[1]:
        proj = pca_proj(flat)
    rgb = flat @ proj.T
    rgb = rgb.reshape(h, w, -1)
    if rgb.shape[2] < 3:
        pad = torch.zeros((h, w, 3 - rgb.shape[2]), device=rgb.device)
        rgb = torch.cat([rgb, pad], dim=2)
    rgb_min = rgb.amin(dim=(0, 1), keepdim=True)
    rgb_max = rgb.amax(dim=(0, 1), keepdim=True)
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)
    if out_hw is not None:
        rgb = rgb.permute(2, 0, 1).unsqueeze(0)
        rgb = F.interpolate(rgb, size=out_hw, mode="nearest")
        rgb = rgb.squeeze(0).permute(1, 2, 0)
    return rgb.clamp(0, 1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="LIBERO demo HDF5 path")
    p.add_argument("--demo", default="demo_0", help="demo key inside /data")
    p.add_argument("--view", default="agentview_rgb", help="RGB key under obs/")
    p.add_argument("--out", default="outputs/anyup_pca.mp4", help="output mp4 path")
    p.add_argument("--max_frames", type=int, default=32, help="max frames to render")
    p.add_argument("--input_size", type=int, default=336, help="resize input frames")
    p.add_argument("--batch", type=int, default=4, help="frames per forward to save memory")
    p.add_argument("--dino", default="vit_base_patch16_dinov3", help="timm model name")
    p.add_argument("--anyup", default="anyup_multi_backbone", help="AnyUp hub name")
    p.add_argument("--anyup_ckpt", default=None, help="local AnyUp checkpoint")
    p.add_argument("--q_chunk_size", type=int, default=None, help="AnyUp query chunk size")
    p.add_argument("--fps", type=int, default=5, help="output fps")
    p.add_argument("--device", default=None, help="cuda or cpu")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    with h5py.File(args.h5, "r") as f:
        demo = f["data"][args.demo]
        video = np.array(demo["obs"][args.view])  # T,H,W,3
    video = video[: args.max_frames]

    vid = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    vid = F.interpolate(vid, size=(args.input_size, args.input_size), mode="bilinear")
    vid_vis = vid.clone()
    vid = (vid - IMAGENET_MEAN) / IMAGENET_STD
    vid = vid.to(device)

    backbone = TimmExtractor(
        model=args.dino,
        pretrained=True,
        frozen=True,
        features="vit_block12",
    ).to(device)
    backbone.eval()

    anyup_dir = Path(__file__).resolve().parents[1] / "anyup"
    if anyup_dir.is_dir():
        upsampler = torch.hub.load(str(anyup_dir), args.anyup, source="local", pretrained=True)
    else:
        upsampler = torch.hub.load("wimmerth/anyup", args.anyup)
    if args.anyup_ckpt:
        state = torch.load(args.anyup_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        upsampler.load_state_dict(state, strict=False)
    upsampler = upsampler.to(device).eval()

    frames = []
    proj_lr = None
    proj_up = None
    for s in range(0, vid.shape[0], args.batch):
        e = min(s + args.batch, vid.shape[0])
        vid_b = vid[s:e]
        vid_vis_b = vid_vis[s:e]
        with torch.no_grad():
            tokens = backbone(vid_b)
            if isinstance(tokens, dict):
                tokens = tokens["vit_block12"]
            if tokens.ndim == 4:
                feats_map = tokens
                h, w = feats_map.shape[-2], feats_map.shape[-1]
                tokens = feats_map.permute(0, 2, 3, 1).reshape(tokens.shape[0], h * w, -1)
            else:
                n = tokens.shape[1]
                img_h = vid_b.shape[-2]
                img_w = vid_b.shape[-1]
                ratio = img_w / img_h
                h = int(round((n / ratio) ** 0.5))
                w = int(round(h * ratio))
                if h * w != n:
                    side = int(round(n**0.5))
                    if side * side != n:
                        raise ValueError(f"Can not infer token grid from {n} tokens")
                    h = side
                    w = side
                feats_map = tokens.view(tokens.shape[0], h, w, -1).permute(0, 3, 1, 2)
            feats_up = upsampler(
                vid_b,
                feats_map,
                output_size=(args.input_size, args.input_size),
                q_chunk_size=args.q_chunk_size,
            )

        if proj_lr is None:
            proj_lr = pca_proj(tokens.reshape(-1, tokens.shape[-1]).to(device))
        if proj_up is None:
            proj_up = pca_proj(
                feats_up.permute(0, 2, 3, 1).reshape(-1, feats_up.shape[1]).to(device)
            )

        for t in range(tokens.shape[0]):
            rgb = vid_vis_b[t].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            dino_map = tokens[t].view(h, w, -1)
            dino_rgb = colorize_map(dino_map, proj_lr, out_hw=(args.input_size, args.input_size))
            up_map = feats_up[t].permute(1, 2, 0)
            up_rgb = colorize_map(up_map, proj_up, out_hw=None)
            frame = np.concatenate([rgb, dino_rgb, up_rgb], axis=1)
            frames.append((frame * 255.0).astype(np.uint8))

    frames = torch.from_numpy(np.stack(frames))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out_path), frames, fps=args.fps)


if __name__ == "__main__":
    main()
