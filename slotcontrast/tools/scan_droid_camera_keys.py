import argparse
from pathlib import Path
from collections import OrderedDict

import h5py
import tqdm


def extract_lab(path: str):
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p.lower() in ("success", "failure") and i > 0:
            return parts[i - 1]
    return None


def list_image_like_keys(h5_path: str):
    keys = []
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                # image-like: 4D and last dim is 1/3/4
                if len(obj.shape) == 4 and obj.shape[-1] in (1, 3, 4):
                    keys.append((name, tuple(obj.shape), str(obj.dtype)))
        f.visititems(visit)
    keys.sort(key=lambda x: x[0])
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="all_episodes.txt (trajectory_im128.h5 paths)")
    ap.add_argument("--out", type=Path, required=True, help="output text file")
    ap.add_argument("--prefer", type=str, default="success",
                    help="prefer picking a success episode first (success|failure|either)")
    args = ap.parse_args()

    lines = [l.strip() for l in args.manifest.read_text().splitlines() if l.strip()]
    # de-dup preserving order
    lines = list(OrderedDict.fromkeys(lines))

    chosen = {}  # lab -> path
    # 2-pass: prefer success then fallback
    for pass_name in (args.prefer, "either"):
        if pass_name == "either":
            want = None
        else:
            want = pass_name.lower()

        for p in lines:
            lab = extract_lab(p)
            if lab is None or lab in chosen:
                continue
            if want is not None:
                parts = Path(p).parts
                if want not in [x.lower() for x in parts]:
                    continue
            chosen[lab] = p

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for lab, p in sorted(chosen.items(), key=lambda x: x[0]):
            out.write(f"=== {lab} ===\n")
            out.write(f"path: {p}\n")
            try:
                keys = list_image_like_keys(p)
                if not keys:
                    out.write("(no image-like 4D datasets found)\n\n")
                    continue
                for k, shape, dtype in keys:
                    out.write(f"{k}\tshape={shape}\tdtype={dtype}\n")
                out.write("\n")
            except Exception as e:
                out.write(f"(failed to read: {e})\n\n")

    print(f"[ok] wrote -> {args.out}")
    print(f"[ok] labs covered: {len(chosen)}")


if __name__ == "__main__":
    main()