#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Re-render Metaworld trajectories into multi-view videos + GT segmentation masks.

This script replays trajectories from an existing Metaworld HF parquet dataset
(`task`, `seed`, `actions`) and renders:
1) third-person RGB video
2) gripper RGB video
3) per-frame GT segmentation mask for each view

The output is written as parquet shards with additional columns:
- `video_third` (HF Video, decode=False payload)
- `video_gripper` (HF Video, decode=False payload)
- `mask_third` (compressed npz bytes, int32 [T, H, W])
- `mask_gripper` (compressed npz bytes, int32 [T, H, W])
- `mask_cat_third` (compressed npz bytes, uint8 [T, H, W], 0:bg 1:fg 2:agent)
- `mask_cat_gripper` (compressed npz bytes, uint8 [T, H, W], 0:bg 1:fg 2:agent)

For compatibility with existing single-view loaders, `video` is also emitted and
set equal to `video_third`.
"""

import argparse
import io
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v2 as imageio


def _ensure_gl_env_early():
    """Set headless GL defaults before importing MuJoCo."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")

    # Avoid permission issues with mesa shader cache on shared clusters.
    if "MESA_SHADER_CACHE_DIR" not in os.environ:
        cache_dir = "/tmp/mesa_shader_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["MESA_SHADER_CACHE_DIR"] = cache_dir


_ensure_gl_env_early()

import mujoco
import numpy as np
from datasets import Dataset, Features, Sequence, Value, Video, load_dataset
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from tqdm import tqdm


OUTPUT_FEATURES = Features(
    {
        "task": Value("string"),
        "seed": Value("int32"),
        "episode": Value("int32"),
        "video": Video(decode=False),
        "video_third": Video(decode=False),
        "video_gripper": Video(decode=False),
        "mask_third": Value("binary"),
        "mask_gripper": Value("binary"),
        "mask_cat_third": Value("binary"),
        "mask_cat_gripper": Value("binary"),
        "states": Sequence(Sequence(Value("float32"))),
        "actions": Sequence(Sequence(Value("float32"))),
        "rewards": Sequence(Value("float32")),
        "camera_third": Value("string"),
        "camera_gripper": Value("string"),
    }
)

BG_NAME_KEYWORDS = (
    "table",
    "floor",
    "ground",
    "wall",
    "arena",
    "workspace",
    "desk",
    "rail",
    "pedestal",
    "torso",
    "controller_box",
    "pedestal_feet",
    "mount",
    "stand",
)
AGENT_NAME_KEYWORDS = (
    "robot",
    "sawyer",
    "franka",
    "arm_base_link",
    "right_l",
    "left_l",
    "wrist",
    "gripper",
    "finger",
    "claw",
    "rightpad",
    "leftpad",
    "pad_geom",
    "endeffector",
    "panda",
)
AGENT_NAME_REGEXES = (
    r"^right_",
    r"^left_",
    r"(^|[_\-])arm($|[_\-])",
    r"(^|[_\-])hand($|[_\-])",
    r"(^|[_\-])eef($|[_\-])",
    r"end_effector",
    r"endeffector",
)

# Valid MuJoCo object-type ids (mjtObj enum values) used in segmentation outputs.
_MJOBJ_TYPE_IDS = {
    int(getattr(mujoco.mjtObj, name))
    for name in dir(mujoco.mjtObj)
    if name.startswith("mjOBJ_")
}
_MJ_SEG_OBJTYPE_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)  # 5 in MuJoCo 3.x
_MJ_SEG_OBJTYPE_SITE = int(mujoco.mjtObj.mjOBJ_SITE)  # 6 in MuJoCo 3.x


def _default_input_dir() -> Optional[Path]:
    root = os.environ.get("JEPAWM_DSET")
    if not root:
        return None
    return Path(root) / "Metaworld" / "data"


def _default_output_dir() -> Optional[Path]:
    root = os.environ.get("JEPAWM_DSET")
    if not root:
        return None
    return Path(root) / "Metaworld_multiview" / "data"


def _task_to_env_id(task_name: str) -> str:
    if not task_name.startswith("mw-"):
        raise ValueError(f"Unexpected Metaworld task format: {task_name}")
    return task_name.split("-", 1)[-1] + "-v3-goal-observable"


def _make_env(task_name: str):
    env_id = _task_to_env_id(task_name)
    if env_id not in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise ValueError(f"Unknown Metaworld env id {env_id} from task {task_name}")
    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=1)
    env.seeded_rand_vec = False
    return env


def _set_camera_position(env, camera_name: str, xyz: Tuple[float, float, float]):
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found.")
    env.model.cam_pos[cam_id] = np.asarray(xyz, dtype=np.float64)


def _available_cameras(env) -> List[str]:
    names = []
    for cam_id in range(env.model.ncam):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        names.append(name if name is not None else f"camera_{cam_id}")
    return names


def _segmentation_to_mask(seg: np.ndarray) -> np.ndarray:
    """Convert MuJoCo segmentation render output to a single int32 label map.

    Segmentation rendering returns [..., 2], but channel order can vary by backend/version.
    We infer the object-type channel robustly, then pack labels while keeping background at 0.
    """
    seg = np.asarray(seg)
    if seg.ndim == 3 and seg.shape[-1] >= 2:
        ch0 = seg[..., 0].astype(np.int32)
        ch1 = seg[..., 1].astype(np.int32)

        # Infer which channel is object type:
        # 1) prefer channel with higher fraction of values in valid mjtObj ids
        # 2) if tied, prefer channel with fewer distinct foreground values
        #    (object types are much fewer than object ids)
        fg0 = ch0[ch0 >= 0]
        fg1 = ch1[ch1 >= 0]

        if fg0.size == 0 or fg1.size == 0:
            obj_type, obj_id = ch0, ch1
        else:
            type_score0 = float(np.mean(np.isin(fg0, list(_MJOBJ_TYPE_IDS))))
            type_score1 = float(np.mean(np.isin(fg1, list(_MJOBJ_TYPE_IDS))))

            if abs(type_score0 - type_score1) > 1e-6:
                if type_score0 > type_score1:
                    obj_type, obj_id = ch0, ch1
                else:
                    obj_type, obj_id = ch1, ch0
            else:
                nuniq0 = np.unique(fg0).size
                nuniq1 = np.unique(fg1).size
                if nuniq0 <= nuniq1:
                    obj_type, obj_id = ch0, ch1
                else:
                    obj_type, obj_id = ch1, ch0

        bg = obj_id < 0
        packed = (obj_type + 1) * 100000 + (obj_id + 1)
        packed[bg] = 0
        return packed.astype(np.int32)
    if seg.ndim == 3 and seg.shape[-1] == 1:
        return seg[..., 0].astype(np.int32)
    return seg.astype(np.int32)


def _render_rgb_and_mask(renderer: mujoco.Renderer, data, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
    # RGB pass
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render().copy()

    # Segmentation pass
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()

    # Keep orientation consistent with existing Metaworld wrapper behavior.
    rgb = rgb[::-1]
    mask = _segmentation_to_mask(seg)[::-1]
    return rgb, mask


def _parse_xyz(xyz_str: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in xyz_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {xyz_str}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _encode_mp4_bytes(frames: np.ndarray, fps: int) -> bytes:
    if frames.dtype != np.uint8:
        raise ValueError(f"RGB frames must be uint8, got {frames.dtype}")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        writer = imageio.get_writer(
            tmp_path,
            format="ffmpeg",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def _encode_mask_npz_bytes(masks: np.ndarray) -> bytes:
    if masks.dtype != np.int32:
        masks = masks.astype(np.int32)
    buf = io.BytesIO()
    np.savez_compressed(buf, mask=masks)
    return buf.getvalue()


def _name_has_keyword(name: Optional[str], keywords: Tuple[str, ...]) -> bool:
    if not name:
        return False
    lower = name.lower()
    return any(k in lower for k in keywords)


def _name_has_regex(name: Optional[str], patterns: Tuple[str, ...]) -> bool:
    if not name:
        return False
    lower = name.lower()
    return any(re.search(p, lower) is not None for p in patterns)


def _is_bg_name(name: Optional[str]) -> bool:
    return _name_has_keyword(name, BG_NAME_KEYWORDS)


def _is_agent_name(name: Optional[str]) -> bool:
    return _name_has_keyword(name, AGENT_NAME_KEYWORDS) or _name_has_regex(
        name, AGENT_NAME_REGEXES
    )


def _pack_label(obj_type: int, obj_id: int) -> int:
    if obj_type < 0 or obj_id < 0:
        return 0
    return int((obj_type + 1) * 100000 + (obj_id + 1))


def _build_category_id_sets(model) -> Tuple[set[int], set[int]]:
    """Return (bg_ids, agent_ids) for packed instance labels for one task model."""
    bg_ids: set[int] = set()
    agent_ids: set[int] = set()
    nbody = int(model.nbody)
    body_parentid = np.asarray(model.body_parentid, dtype=np.int32)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in range(nbody)]

    # Use body-level labels first; this is more robust than geom/site name-only rules.
    agent_body = np.zeros(nbody, dtype=bool)
    bg_body = np.zeros(nbody, dtype=bool)
    for bid, bname in enumerate(body_names):
        if _is_agent_name(bname):
            agent_body[bid] = True
        if _is_bg_name(bname):
            bg_body[bid] = True

    # Body ids are topologically ordered in MuJoCo models (parent id < child id).
    # Propagate labels along the tree so unnamed child geoms inherit body class.
    for bid in range(nbody):
        pid = int(body_parentid[bid])
        if pid >= 0:
            if agent_body[pid]:
                agent_body[bid] = True
            if bg_body[pid]:
                bg_body[bid] = True

    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        bid = int(model.geom_bodyid[gid])
        packed = _pack_label(_MJ_SEG_OBJTYPE_GEOM, gid)
        if (0 <= bid < nbody and bg_body[bid]) or _is_bg_name(gname):
            bg_ids.add(packed)
        if (0 <= bid < nbody and agent_body[bid]) or _is_agent_name(gname):
            agent_ids.add(packed)

    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        bid = int(model.site_bodyid[sid])
        packed = _pack_label(_MJ_SEG_OBJTYPE_SITE, sid)
        if (0 <= bid < nbody and bg_body[bid]) or _is_bg_name(sname):
            bg_ids.add(packed)
        if (0 <= bid < nbody and agent_body[bid]) or _is_agent_name(sname):
            agent_ids.add(packed)

    # If an element matches both classes, prefer agent assignment (robot safety).
    bg_ids.difference_update(agent_ids)
    return bg_ids, agent_ids


def _instance_to_category_mask(
    instance_mask: np.ndarray, bg_ids: set[int], agent_ids: set[int]
) -> np.ndarray:
    """Map packed instance ids to 3 classes: 0=bg, 1=fg, 2=agent."""
    m = instance_mask.astype(np.int32, copy=False)
    out = np.ones(m.shape, dtype=np.uint8)
    bg = (m == 0) | np.isin(m, list(bg_ids))
    agent = np.isin(m, list(agent_ids))
    out[bg] = 0
    out[agent & (~bg)] = 2
    return out


def _normalize_packed_instance_mask(instance_mask: np.ndarray) -> np.ndarray:
    """Normalize packed instance ids to canonical layout.

    Canonical packed layout is:
      packed = (obj_type + 1) * 100000 + (obj_id + 1)

    Some previously exported masks may have swapped layout:
      packed_swapped = (obj_id + 1) * 100000 + (obj_type + 1)

    This function detects swapped layout using valid `mjtObj` type ids and, if needed,
    rewrites the mask to canonical layout.
    """
    m = instance_mask.astype(np.int32, copy=False)
    fg = m > 0
    if not np.any(fg):
        return m

    packed_fg = m[fg]
    type_from_high = (packed_fg // 100000) - 1
    type_from_low = (packed_fg % 100000) - 1

    score_high = float(np.mean(np.isin(type_from_high, list(_MJOBJ_TYPE_IDS))))
    score_low = float(np.mean(np.isin(type_from_low, list(_MJOBJ_TYPE_IDS))))

    # Already canonical (or ambiguous): keep as is.
    if score_high >= score_low:
        return m

    obj_type = (m % 100000) - 1
    obj_id = (m // 100000) - 1
    repaired = (obj_type + 1) * 100000 + (obj_id + 1)
    repaired[~fg] = 0
    return repaired.astype(np.int32, copy=False)


def _iter_selected_indices(ds, start: int, end: Optional[int], max_episodes: Optional[int]) -> List[int]:
    n = len(ds)
    start = max(0, int(start))
    stop = n if end is None else min(n, int(end))
    if stop <= start:
        return []
    indices = list(range(start, stop))
    if max_episodes is not None:
        indices = indices[: int(max_episodes)]
    return indices


def _to_serializable_rows(rows: Iterable[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        out.append(
            {
                "task": str(row["task"]),
                "seed": int(row["seed"]),
                "episode": int(row["episode"]),
                "video": {"bytes": row["video"], "path": None},
                "video_third": {"bytes": row["video_third"], "path": None},
                "video_gripper": {"bytes": row["video_gripper"], "path": None},
                "mask_third": row["mask_third"],
                "mask_gripper": row["mask_gripper"],
                "mask_cat_third": row["mask_cat_third"],
                "mask_cat_gripper": row["mask_cat_gripper"],
                "states": row["states"],
                "actions": row["actions"],
                "rewards": row["rewards"],
                "camera_third": str(row["camera_third"]),
                "camera_gripper": str(row["camera_gripper"]),
            }
        )
    return out


def _write_shard(rows: List[Dict], out_path: Path):
    ds = Dataset.from_list(_to_serializable_rows(rows), features=OUTPUT_FEATURES)
    ds.to_parquet(str(out_path))


def _ensure_gl_env():
    # Kept for explicit call in main; import-time defaults are set in _ensure_gl_env_early().
    _ensure_gl_env_early()


def main():
    parser = argparse.ArgumentParser(
        description="Re-render Metaworld into third+gripper views with GT segmentation masks."
    )
    parser.add_argument("--input-dir", type=Path, default=_default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/metaworld_hf_cache"))
    parser.add_argument("--third-camera", type=str, default="corner2")
    parser.add_argument("--gripper-camera", type=str, default="gripperPOV")
    parser.add_argument(
        "--third-cam-pos",
        type=str,
        default="0.75,0.075,0.7",
        help="Third camera xyz override. Use 'none' to disable. Default matches repo Metaworld wrapper.",
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="How many actions to apply before capturing first frame. 0 means capture reset frame first.",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        raise ValueError("Input directory is required. Pass --input-dir or set JEPAWM_DSET.")
    if args.output_dir is None:
        raise ValueError("Output directory is required. Pass --output-dir or set JEPAWM_DSET.")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.frame_offset < 0:
        raise ValueError("--frame-offset must be >= 0")

    _ensure_gl_env()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading source dataset from: {args.input_dir}")
    src = load_dataset("parquet", data_dir=str(args.input_dir), split="train", cache_dir=str(args.cache_dir))
    src = src.cast_column("video", Video(decode=False))
    print(f"Total source episodes: {len(src)}")

    indices = _iter_selected_indices(src, args.start_index, args.end_index, args.max_episodes)
    if not indices:
        raise ValueError("No episodes selected. Check --start-index/--end-index/--max-episodes.")

    n_shards = math.ceil(len(indices) / args.shard_size)
    print(
        f"Selected episodes: {len(indices)} | shard_size: {args.shard_size} | "
        f"output shards: {n_shards}"
    )
    print(
        f"Cameras: third={args.third_camera}, gripper={args.gripper_camera} | "
        f"resolution={args.width}x{args.height} | fps={args.fps}"
    )
    print(f"Frame alignment: frame_offset={args.frame_offset}")
    if str(args.third_cam_pos).lower() != "none":
        print(f"Third camera pose override: {args.third_camera} -> {args.third_cam_pos}")
        third_cam_pos_xyz = _parse_xyz(args.third_cam_pos)
    else:
        print("Third camera pose override: disabled")
        third_cam_pos_xyz = None
    print(
        "GL backend: "
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
        f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}"
    )

    env_by_task: Dict[str, object] = {}
    renderer_by_task: Dict[str, mujoco.Renderer] = {}
    category_ids_by_task: Dict[str, Tuple[set[int], set[int]]] = {}

    try:
        for shard_idx in range(n_shards):
            s = shard_idx * args.shard_size
            e = min((shard_idx + 1) * args.shard_size, len(indices))
            shard_indices = indices[s:e]
            out_path = args.output_dir / f"train-{shard_idx:05d}-of-{n_shards:05d}.parquet"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] shard exists: {out_path}")
                continue

            rows_out: List[Dict] = []
            pbar = tqdm(shard_indices, desc=f"shard {shard_idx + 1}/{n_shards}", leave=False)
            for k, row_idx in enumerate(pbar):
                try:
                    row = src[int(row_idx)]
                    task = str(row["task"])
                    seed = int(row["seed"])
                    episode = int(row["episode"])
                    actions = np.asarray(row["actions"], dtype=np.float32)

                    env = env_by_task.get(task)
                    renderer = renderer_by_task.get(task)
                    if env is None or renderer is None:
                        env = _make_env(task)
                        env_by_task[task] = env
                        category_ids_by_task[task] = _build_category_id_sets(env.model)
                        if third_cam_pos_xyz is not None:
                            _set_camera_position(env, args.third_camera, third_cam_pos_xyz)
                        try:
                            renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
                        except Exception as exc:
                            raise RuntimeError(
                                "Failed to initialize MuJoCo offscreen renderer. "
                                "Set MUJOCO_GL=egl (or MUJOCO_GL=osmesa) before running, "
                                "and verify EGL/OSMesa is available on this server."
                            ) from exc
                        renderer_by_task[task] = renderer

                        cams = set(_available_cameras(env))
                        missing = [
                            cam
                            for cam in [args.third_camera, args.gripper_camera]
                            if cam not in cams
                        ]
                        if missing:
                            raise ValueError(
                                f"Task {task} missing camera(s) {missing}. "
                                f"Available cameras: {sorted(cams)}"
                            )

                    env.reset(seed=seed)
                    n_target_frames = len(row["states"])
                    action_cursor = 0
                    for _ in range(args.frame_offset):
                        if action_cursor >= len(actions):
                            break
                        env.step(actions[action_cursor])
                        action_cursor += 1

                    third_rgbs = []
                    grip_rgbs = []
                    third_masks = []
                    grip_masks = []

                    for _ in range(n_target_frames):
                        third_rgb, third_mask = _render_rgb_and_mask(renderer, env.data, args.third_camera)
                        grip_rgb, grip_mask = _render_rgb_and_mask(renderer, env.data, args.gripper_camera)
                        third_rgbs.append(third_rgb)
                        grip_rgbs.append(grip_rgb)
                        third_masks.append(third_mask)
                        grip_masks.append(grip_mask)
                        if action_cursor < len(actions):
                            env.step(actions[action_cursor])
                            action_cursor += 1

                    third_rgbs = np.stack(third_rgbs, axis=0)
                    grip_rgbs = np.stack(grip_rgbs, axis=0)
                    third_masks = np.stack(third_masks, axis=0).astype(np.int32)
                    grip_masks = np.stack(grip_masks, axis=0).astype(np.int32)
                    bg_ids, agent_ids = category_ids_by_task[task]
                    third_cat_masks = _instance_to_category_mask(third_masks, bg_ids, agent_ids)
                    grip_cat_masks = _instance_to_category_mask(grip_masks, bg_ids, agent_ids)

                    third_video_bytes = _encode_mp4_bytes(third_rgbs, fps=args.fps)
                    grip_video_bytes = _encode_mp4_bytes(grip_rgbs, fps=args.fps)
                    third_mask_bytes = _encode_mask_npz_bytes(third_masks)
                    grip_mask_bytes = _encode_mask_npz_bytes(grip_masks)
                    third_cat_mask_bytes = _encode_mask_npz_bytes(third_cat_masks.astype(np.int32))
                    grip_cat_mask_bytes = _encode_mask_npz_bytes(grip_cat_masks.astype(np.int32))

                    rows_out.append(
                        {
                            "task": task,
                            "seed": seed,
                            "episode": episode,
                            "video": third_video_bytes,
                            "video_third": third_video_bytes,
                            "video_gripper": grip_video_bytes,
                            "mask_third": third_mask_bytes,
                            "mask_gripper": grip_mask_bytes,
                            "mask_cat_third": third_cat_mask_bytes,
                            "mask_cat_gripper": grip_cat_mask_bytes,
                            "states": row["states"],
                            "actions": row["actions"],
                            "rewards": row["rewards"],
                            "camera_third": args.third_camera,
                            "camera_gripper": args.gripper_camera,
                        }
                    )

                    if (k + 1) % args.log_every == 0:
                        pbar.set_postfix(task=task, seed=seed, kept=len(rows_out))
                except Exception as exc:
                    if args.skip_failed:
                        print(f"[warn] row {row_idx} failed: {exc}")
                        continue
                    raise

            if not rows_out:
                print(f"[warn] no successful rows in shard {shard_idx}, skipping write.")
                continue

            _write_shard(rows_out, out_path)
            print(f"[ok] wrote {len(rows_out)} episodes -> {out_path}")
    finally:
        for renderer in renderer_by_task.values():
            try:
                renderer.close()
            except Exception:
                pass
        for env in env_by_task.values():
            try:
                env.close()
            except Exception:
                pass

    print("Done.")


if __name__ == "__main__":
    main()
