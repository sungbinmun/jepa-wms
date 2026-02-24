#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

try:
    import imageio.v3 as iio
except Exception:
    iio = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Export exactly one 4s clip per lab from exported DROID H5 to MP4 (exo + hand)."
    )
    p.add_argument("--h5", required=True, type=Path, help="Exported H5 path")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory for mp4s")
    p.add_argument("--fps", type=float, default=15.0, help="MP4 fps (default 15)")

    p.add_argument(
        "--exo-view-key",
        type=str,
        default="observation__camera__image__varied_camera_1_left_image",
        help="View dataset name under clips/<clip_id>/frames/ for EXO mp4",
    )
    p.add_argument(
        "--hand-view-key",
        type=str,
        default="observation__camera__image__hand_camera_left_image",
        help="View dataset name under clips/<clip_id>/frames/ for HAND mp4",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="If set, skip labs where either view is missing. Otherwise export what exists.",
    )
    p.add_argument("--max-labs", type=int, default=None, help="Optional cap on labs (debug)")
    return p.parse_args()


def write_mp4(frames_thwc_u8: np.ndarray, out_path: Path, fps: float):
    if iio is None:
        raise RuntimeError("Need imageio w/ ffmpeg. Install: pip install 'imageio[ffmpeg]'")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = frames_thwc_u8
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    # imageio expects (T,H,W,3) uint8
    iio.imwrite(str(out_path), frames, fps=fps)


def parse_lab_from_clip_id(clip_id: str) -> Tuple[Optional[str], Optional[str], str]:
    # clip_id example: success__AUTOLab__0000000
    parts = clip_id.split("__")
    if len(parts) >= 3:
        outcome, lab, short_id = parts[0], parts[1], "__".join(parts[2:])
        return outcome, lab, short_id
    return None, None, clip_id


def load_view(clip_g: h5py.Group, view_key: str) -> Optional[np.ndarray]:
    if "frames" not in clip_g:
        return None
    frames_g = clip_g["frames"]
    if view_key not in frames_g:
        return None
    return np.array(frames_g[view_key][...])  # (T,H,W,3)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        if "clips" not in f:
            raise KeyError("H5 missing top-level group 'clips'")
        clips_g = f["clips"]

        # lab -> first clip_id
        first_clip_for_lab: Dict[str, str] = {}
        for clip_id in clips_g.keys():
            _, lab, _ = parse_lab_from_clip_id(clip_id)
            if lab is None:
                continue
            if lab not in first_clip_for_lab:
                first_clip_for_lab[lab] = clip_id

        labs = sorted(first_clip_for_lab.keys())
        if args.max_labs is not None:
            labs = labs[: args.max_labs]

        print(f"[info] Found {len(first_clip_for_lab)} labs. Exporting {len(labs)} labs...")

        for lab in labs:
            clip_id = first_clip_for_lab[lab]
            clip_g = clips_g[clip_id]

            exo = load_view(clip_g, args.exo_view_key)
            hand = load_view(clip_g, args.hand_view_key)

            if args.skip_missing and (exo is None or hand is None):
                print(
                    f"[warn] {lab}: missing view(s) in {clip_id} "
                    f"(exo={'ok' if exo is not None else 'None'}, hand={'ok' if hand is not None else 'None'}) "
                    f"(skipping lab)"
                )
                continue

            lab_dir = args.outdir / lab
            lab_dir.mkdir(parents=True, exist_ok=True)

            if exo is not None:
                out_exo = lab_dir / f"{clip_id}__EXO.mp4"
                write_mp4(exo, out_exo, fps=args.fps)
                print(f"[ok] {lab}: wrote EXO  -> {out_exo}")
            else:
                print(f"[warn] {lab}: EXO view missing for {clip_id}")

            if hand is not None:
                out_hand = lab_dir / f"{clip_id}__HAND.mp4"
                write_mp4(hand, out_hand, fps=args.fps)
                print(f"[ok] {lab}: wrote HAND -> {out_hand}")
            else:
                print(f"[warn] {lab}: HAND view missing for {clip_id}")


if __name__ == "__main__":
    main()