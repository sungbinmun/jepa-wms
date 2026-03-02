"""Extract Metaworld slots from a SlotContrast checkpoint for C-JEPA training.

Outputs 4 aligned pickle files with split dictionaries:
- slots.pkl   : {'train': {key: [T, S, D]}, 'val': {...}}
- actions.pkl : {'train': {key: [T-1, A]},   'val': {...}}
- proprio.pkl : {'train': {key: [T, P]},     'val': {...}}
- states.pkl  : {'train': {key: [T, X]},     'val': {...}}
"""

from __future__ import annotations

import argparse
import inspect
import io
import os
import pickle as pkl
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


def _default_parquet_dir() -> Path:
    root = os.environ.get("JEPAWM_DSET")
    if not root:
        raise ValueError("JEPAWM_DSET is not set. Pass --parquet-dir explicitly.")
    return Path(root) / "Metaworld_multiview_full_merged" / "data"


def _decode_video_bytes(video_bytes: bytes) -> np.ndarray:
    reader = imageio.get_reader(io.BytesIO(video_bytes), format="mp4")
    frames = [frame for frame in reader]
    reader.close()
    if not frames:
        raise ValueError("Decoded empty video.")
    return np.stack(frames, axis=0)  # [T, H, W, C]


def _parse_indices(spec: str) -> np.ndarray:
    values = [int(v.strip()) for v in spec.split(",") if v.strip()]
    if not values:
        raise ValueError("proprio_indices cannot be empty.")
    return np.asarray(values, dtype=np.int64)


def _torch_load_trusted(path: str, map_location: str | torch.device):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _load_slotcontrast_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
    repo_root: Path,
):
    slotcontrast_root = repo_root / "slotcontrast"
    if str(slotcontrast_root) not in sys.path:
        sys.path.insert(0, str(slotcontrast_root))

    from slotcontrast import configuration, models
    from slotcontrast.data.transforms import build as build_transforms

    config = configuration.load_config(str(config_path))
    model = models.build(config.model, config.optimizer)
    ckpt = _torch_load_trusted(str(checkpoint_path), "cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a Lightning checkpoint containing `state_dict`.")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)

    train_tf_cfg = config.dataset.train_pipeline.transforms
    tf_map = build_transforms(train_tf_cfg)
    if "video" not in tf_map:
        raise ValueError("SlotContrast transform map does not contain `video` transform.")
    video_tf = tf_map["video"]

    return model, video_tf, config


def _infer_config_path(checkpoint_path: Path) -> Path:
    # .../<run_dir>/checkpoints/step=XXXX.ckpt -> .../<run_dir>/settings.yaml
    candidate = checkpoint_path.parent.parent / "settings.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not infer config path from checkpoint. Expected: {candidate}. "
            "Pass --slotcontrast-config explicitly."
        )
    return candidate


def _extract_slots_for_video(
    model,
    video_tf,
    frames: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    video = video_tf(frames).unsqueeze(0).to(device)  # [1, T, C, H, W]
    with torch.no_grad():
        enc_out = model.encoder(video)
        features = enc_out["features"]
        slots_init = model.initializer(batch_size=1).to(device)
        proc_out = model.processor(slots_init, features)
        slots = proc_out["state"][0].detach().cpu().float().numpy()
    return slots


def main():
    parser = argparse.ArgumentParser(description="Extract Metaworld slots using SlotContrast checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to SlotContrast .ckpt file.")
    parser.add_argument(
        "--slotcontrast-config",
        type=Path,
        default=None,
        help="Path to SlotContrast training config/settings.yaml. "
        "If omitted, inferred as <checkpoint_parent_parent>/settings.yaml.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Metaworld parquet directory. Default: $JEPAWM_DSET/Metaworld_multiview_full_merged/data",
    )
    parser.add_argument("--video-key", type=str, default="video_gripper", choices=["video", "video_third", "video_gripper"])
    parser.add_argument("--train-fraction", type=float, default=0.99)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--proprio-indices", type=str, default="0,1,2,3")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output prefix path without extension.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = (
        args.slotcontrast_config.expanduser().resolve()
        if args.slotcontrast_config is not None
        else _infer_config_path(checkpoint_path)
    )
    if not config_path.exists():
        raise FileNotFoundError(f"SlotContrast config not found: {config_path}")

    parquet_dir = (
        args.parquet_dir.expanduser().resolve() if args.parquet_dir is not None else _default_parquet_dir()
    )
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    proprio_idx = _parse_indices(args.proprio_indices)

    repo_root = Path(__file__).resolve().parents[3]
    model, video_tf, _ = _load_slotcontrast_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
        repo_root=repo_root,
    )

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")

    rows_per_file = [pq.ParquetFile(str(pf)).metadata.num_rows for pf in parquet_files]
    total_rows = int(sum(rows_per_file))
    if total_rows <= 1:
        raise ValueError(f"Too few rows in dataset: {total_rows}")

    global_indices = np.arange(total_rows, dtype=np.int64)
    rng = np.random.RandomState(args.split_seed)
    rng.shuffle(global_indices)
    split_at = int(round(total_rows * args.train_fraction))
    split_at = min(max(split_at, 1), total_rows - 1)
    is_train = np.zeros(total_rows, dtype=bool)
    is_train[global_indices[:split_at]] = True

    outputs = {
        "slots": {"train": {}, "val": {}},
        "actions": {"train": {}, "val": {}},
        "proprio": {"train": {}, "val": {}},
        "states": {"train": {}, "val": {}},
    }

    cols = [args.video_key, "task", "episode", "actions", "states"]

    global_row_idx = 0
    processed = 0
    pbar = tqdm(total=total_rows, desc="Extracting slots")
    for pf in parquet_files:
        table = pq.read_table(str(pf), columns=cols, use_threads=True)
        data = table.to_pydict()
        n_rows = len(data["episode"])
        for i in range(n_rows):
            if args.max_rows is not None and processed >= args.max_rows:
                break

            split = "train" if is_train[global_row_idx] else "val"
            key = f"metaworld_{split}_{global_row_idx}"
            global_row_idx += 1
            processed += 1
            pbar.update(1)

            video_obj = data[args.video_key][i]
            if not isinstance(video_obj, dict) or video_obj.get("bytes") is None:
                continue
            frames = _decode_video_bytes(video_obj["bytes"])
            slots = _extract_slots_for_video(model, video_tf, frames, device=device).astype(np.float32)

            actions = np.asarray(data["actions"][i], dtype=np.float32)
            states = np.asarray(data["states"][i], dtype=np.float32)
            if states.ndim != 2:
                continue
            if proprio_idx.max() >= states.shape[1]:
                raise ValueError(
                    f"Requested proprio index {int(proprio_idx.max())} out of state dim {states.shape[1]}"
                )
            proprio = states[:, proprio_idx]

            # Keep all modalities time-aligned for PushTSlotDataset assumptions.
            t_common = int(min(slots.shape[0], actions.shape[0], proprio.shape[0], states.shape[0]))
            if t_common <= 1:
                continue
            slots = slots[:t_common]
            actions = actions[:t_common]
            proprio = proprio[:t_common]
            states = states[:t_common]

            outputs["slots"][split][key] = slots
            outputs["actions"][split][key] = actions
            outputs["proprio"][split][key] = proprio.astype(np.float32)
            outputs["states"][split][key] = states.astype(np.float32)
        if args.max_rows is not None and processed >= args.max_rows:
            break
    pbar.close()

    if not outputs["slots"]["train"] or not outputs["slots"]["val"]:
        raise RuntimeError("Extraction produced empty train/val splits.")

    out_prefix = args.output_prefix.expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "slots": out_prefix.with_name(f"{out_prefix.name}_slots.pkl"),
        "actions": out_prefix.with_name(f"{out_prefix.name}_actions.pkl"),
        "proprio": out_prefix.with_name(f"{out_prefix.name}_proprio.pkl"),
        "states": out_prefix.with_name(f"{out_prefix.name}_states.pkl"),
    }
    for name, path in paths.items():
        with open(path, "wb") as f:
            pkl.dump(outputs[name], f)
        print(f"[saved] {name}: {path}")


if __name__ == "__main__":
    main()
