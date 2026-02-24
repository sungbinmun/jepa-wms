#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Inspect MuJoCo segmentation ids used for Metaworld category masks.

This utility reports:
- MuJoCo segmentation object-type ids (geom/site)
- background packed id convention (0)
- per-task packed ids mapped to category (background/agent)

Packed id format used by this repo:
  packed = (obj_type + 1) * 100000 + (obj_id + 1)

Background pixels from MuJoCo segmentation are (obj_type=-1, obj_id=-1),
which map to packed id 0.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Optional

import mujoco
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

import preprocess_metaworld_multiview as pm


def _task_to_env_id(task: str) -> str:
    if not task.startswith("mw-"):
        raise ValueError(f"Expected task format mw-*, got: {task}")
    return task.split("-", 1)[-1] + "-v3-goal-observable"


def _packed_to_desc(model, packed_id: int) -> str:
    if packed_id <= 0:
        return "raw background"
    obj_type = (packed_id // 100000) - 1
    obj_id = (packed_id % 100000) - 1

    if obj_type == int(mujoco.mjtObj.mjOBJ_GEOM):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, obj_id)
        if obj_id < 0 or obj_id >= model.ngeom:
            return f"type=GEOM(5) id={obj_id} <out-of-range>"
        bid = int(model.geom_bodyid[obj_id])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        return f"type=GEOM(5) id={obj_id} geom={gname} body={bname}"

    if obj_type == int(mujoco.mjtObj.mjOBJ_SITE):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, obj_id)
        if obj_id < 0 or obj_id >= model.nsite:
            return f"type=SITE(6) id={obj_id} <out-of-range>"
        bid = int(model.site_bodyid[obj_id])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        return f"type=SITE(6) id={obj_id} site={sname} body={bname}"

    return f"type={obj_type} id={obj_id}"


def _print_ids(title: str, model, ids: Iterable[int], max_ids: int):
    ids = sorted(int(x) for x in ids)
    print(f"{title}: {len(ids)} ids")
    for packed_id in ids[:max_ids]:
        print(f"  {packed_id}: {_packed_to_desc(model, packed_id)}")
    if len(ids) > max_ids:
        print(f"  ... ({len(ids) - max_ids} more)")


def _inspect_task(task: str, max_ids: int):
    env_id = _task_to_env_id(task)
    if env_id not in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise ValueError(f"Unknown task: {task} -> {env_id}")

    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=1)
    env.seeded_rand_vec = False
    try:
        bg_ids, agent_ids = pm._build_category_id_sets(env.model)
        print(f"\n=== {task} ===")
        _print_ids("background", env.model, bg_ids, max_ids=max_ids)
        _print_ids("agent", env.model, agent_ids, max_ids=max_ids)
    finally:
        env.close()


def _normalize_tasks(task_args: Optional[List[str]], all_tasks: bool) -> List[str]:
    if all_tasks:
        out = []
        for env_id in sorted(ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE):
            out.append("mw-" + env_id.replace("-v3-goal-observable", ""))
        return out
    if task_args:
        return task_args
    return ["mw-assembly"]


def main():
    parser = argparse.ArgumentParser(description="Inspect Metaworld segmentation ids for bg/agent categories.")
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task name (mw-*). Can be passed multiple times. Default: mw-assembly.",
    )
    parser.add_argument("--all-tasks", action="store_true", help="Inspect all Metaworld tasks.")
    parser.add_argument(
        "--max-ids",
        type=int,
        default=40,
        help="Max ids to print per category/task.",
    )
    args = parser.parse_args()

    print("MuJoCo segmentation object-type ids:")
    print(f"  GEOM: {int(mujoco.mjtObj.mjOBJ_GEOM)}")
    print(f"  SITE: {int(mujoco.mjtObj.mjOBJ_SITE)}")
    print("Raw segmentation background pixel: (obj_type=-1, obj_id=-1)")
    print("Packed background id used by this repo: 0")

    tasks = _normalize_tasks(args.task, args.all_tasks)
    for task in tasks:
        _inspect_task(task, max_ids=max(1, int(args.max_ids)))


if __name__ == "__main__":
    main()
