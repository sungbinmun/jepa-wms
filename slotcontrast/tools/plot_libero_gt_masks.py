import argparse

import h5py
import imageio
import numpy as np
import torch

from slotcontrast.visualizations import color_map, create_grid_frame_rgb


def parse_ids(text: str):
    if text is None:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def ensure_even_size(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return frame


def mask_image(image_uint8: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    img = image_uint8.float() / 255.0
    mask_f = mask.float()
    if mask_f.ndim == 2:
        mask_f = mask_f.unsqueeze(0)
    mask_f = mask_f.clamp(0.0, 1.0)
    return (img * mask_f).permute(1, 2, 0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="LIBERO H5 path")
    p.add_argument("--demo", default="demo_0", help="demo key under /data")
    p.add_argument("--view", default="agentview_rgb", help="obs view key")
    p.add_argument("--out", default="outputs/libero_gt_masks.mp4", help="output mp4 path")
    p.add_argument("--max_frames", type=int, default=64, help="max frames to render")
    p.add_argument("--fps", type=int, default=10, help="output fps")
    p.add_argument("--alpha", type=float, default=0.5, help="mask overlay alpha")
    p.add_argument("--bg_ids", default="0", help="comma-separated background ids")
    p.add_argument("--agent_ids", default="65535", help="comma-separated agent ids")
    args = p.parse_args()

    bg_ids = set(parse_ids(args.bg_ids))
    agent_ids = set(parse_ids(args.agent_ids))

    with h5py.File(args.h5, "r") as f:
        root = f["data"] if "data" in f else f
        if args.demo not in root:
            raise KeyError(f"demo '{args.demo}' not found. keys: {list(root.keys())[:5]}")
        demo = root[args.demo]
        obs = demo["obs"]
        if args.view not in obs:
            raise KeyError(f"view '{args.view}' not found. keys: {list(obs.keys())}")
        video = np.array(obs[args.view])  # (T,H,W,3) uint8
        seg = np.array(demo["segmentation"])  # (T,H,W) uint16

    video = video[: args.max_frames]
    seg = seg[: args.max_frames]

    if seg.ndim == 4 and seg.shape[-1] == 1:
        seg = seg[..., 0]

    obj_ids = sorted(
        [int(x) for x in np.unique(seg) if int(x) not in bg_ids and int(x) not in agent_ids]
    )
    all_colors = color_map(2 + len(obj_ids))

    frames = []
    for t in range(video.shape[0]):
        frame = video[t]
        frame_t = torch.from_numpy(frame).permute(2, 0, 1)
        seg_t = torch.from_numpy(seg[t])

        bg_mask = torch.zeros_like(seg_t, dtype=torch.bool)
        for idx in bg_ids:
            bg_mask |= seg_t == idx
        agent_mask = torch.zeros_like(seg_t, dtype=torch.bool)
        for idx in agent_ids:
            agent_mask |= seg_t == idx
        obj_masks = [(seg_t == idx) for idx in obj_ids]

        masks_all = torch.stack([bg_mask, agent_mask] + obj_masks, dim=0)
        union_all = masks_all.any(dim=0)
        union_obj = torch.stack(obj_masks, dim=0).any(dim=0) if obj_masks else torch.zeros_like(bg_mask)

        panels = [
            frame.astype(np.float32) / 255.0,
            mask_image(frame_t, union_all),
            mask_image(frame_t, bg_mask),
            mask_image(frame_t, agent_mask),
            mask_image(frame_t, union_obj),
        ]

        grid = create_grid_frame_rgb(
            panels, grid_size=(1, len(panels)), image_size=frame.shape[:2], padding=2
        )
        frames.append((grid * 255).astype(np.uint8))

    out_dir = "/".join(args.out.split("/")[:-1])
    if out_dir:
        import os

        os.makedirs(out_dir, exist_ok=True)
    with imageio.get_writer(args.out, fps=args.fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(ensure_even_size(frame))


if __name__ == "__main__":
    main()
