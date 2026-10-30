"""Find a ring-finger side contact on the movable face layer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from validate_poses import (
    PENETRATION_LIMIT,
    ROOT,
    actuator_id,
    finger_prefix,
    geom_name,
    hand_geoms,
    load_pose,
    object_id,
    qpos_address,
    sha256,
)


MAIN_SCENE = ROOT / "scenes" / "table_cube.xml"
SCENE = ROOT / "scenes" / "table_articulated_face_test.xml"
SOURCE_POSE = ROOT / "poses" / "articulated_grasp_R01.json"
OUTPUT_POSE = ROOT / "poses" / "articulated_face_reach_RF02.json"
OUTPUT_LOG = ROOT / "logs" / "prompt24_articulated_face_reach_planner_revision.txt"
EXPECTED_MAIN_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
EXPECTED_SCENE_SHA256 = "baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb"
SEED = 24001
SAMPLES = 20_000
TRANSITION_STEPS = 300
TARGET_DEPTH_M = -0.0010
RING_JOINTS = ("rf_mcp", "rf_rot", "rf_pip", "rf_dip")


def distal(name: str) -> bool:
    return "_md_" in name or "_ds_" in name or name.endswith("_tip")


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before reachability planning")
    payload = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    mount, joints, _, _ = load_pose(SOURCE_POSE, model)
    data = mujoco.MjData(model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_qpos = int(model.jnt_qposadr[face_joint])
    data.qpos[cube_qpos : cube_qpos + 7] = (
        *payload["cube_start_xyz"],
        *payload["cube_start_quat_wxyz"],
    )
    data.qpos[face_qpos] = float(payload["face_turn_rad"])
    for name, value in mount.items() | joints.items():
        data.qpos[qpos_address(model, name)] = value
        data.ctrl[actuator_id(model, name)] = value
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    for step in range(800):
        phase = min((step + 1) / 300, 1.0)
        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
        data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
        mujoco.mj_step(model, data)

    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    hand = hand_geoms(model)
    hand_set = set(hand)
    joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RING_JOINTS]
    lower = np.array([model.jnt_range[joint, 0] for joint in joint_ids])
    upper = np.array([model.jnt_range[joint, 1] for joint in joint_ids])
    base_qpos = data.qpos.copy()
    rng = np.random.default_rng(SEED)
    samples = rng.uniform(lower, upper, size=(SAMPLES, len(RING_JOINTS)))
    rows: list[tuple[float, np.ndarray, list[str], float]] = []
    rejected_penetration = 0
    rejected_support = 0
    rejected_core_jam = 0
    started = time.monotonic()

    for values in samples:
        data.qpos[:] = base_qpos
        for name, value in zip(RING_JOINTS, values, strict=True):
            data.qpos[qpos_address(model, name)] = value
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        deepest = 0.0
        core_prefixes: set[str] = set()
        ring_core = False
        side_geoms: set[str] = set()
        side_depths: list[float] = []
        rotation = data.xmat[face_body].reshape(3, 3)
        center = data.xpos[face_body]
        for contact in data.contact[: data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            pair = {first, second}
            if core_geom in pair:
                other = second if first == core_geom else first
                if other in hand_set:
                    deepest = min(deepest, float(contact.dist))
                    prefix = finger_prefix(geom_name(model, other))
                    if prefix:
                        core_prefixes.add(prefix)
                    ring_core |= prefix == "rf"
            elif face_geom in pair:
                other = second if first == face_geom else first
                if other in hand_set:
                    deepest = min(deepest, float(contact.dist))
                    name = geom_name(model, other)
                    local = rotation.T @ (np.asarray(contact.pos) - center)
                    on_side = max(abs(local[0]), abs(local[1])) >= 0.023 and abs(local[2]) <= 0.0085
                    if finger_prefix(name) == "rf" and distal(name) and on_side:
                        side_geoms.add(name)
                        side_depths.append(float(contact.dist))
            elif first in hand_set and second in hand_set:
                first_prefix = finger_prefix(geom_name(model, first))
                second_prefix = finger_prefix(geom_name(model, second))
                if first_prefix and second_prefix and first_prefix != second_prefix:
                    deepest = min(deepest, float(contact.dist))
        if deepest < -PENETRATION_LIMIT:
            rejected_penetration += 1
            continue
        if not {"th", "if", "mf"}.issubset(core_prefixes):
            rejected_support += 1
            continue
        if ring_core:
            rejected_core_jam += 1
            continue
        if side_depths:
            target_depth = max(side_depths)
            score = -abs(target_depth - TARGET_DEPTH_M)
            rows.append((score, values.copy(), sorted(side_geoms), deepest))

    rows.sort(key=lambda row: row[0], reverse=True)
    elapsed = time.monotonic() - started
    best = rows[0] if rows else None
    if best is not None:
        _, values, side_geoms, deepest = best
        candidate = {
            "test_only": True,
            "label": "articulated_face_reach_RF01",
            "source_held_pose": str(SOURCE_POSE),
            "transition_steps": TRANSITION_STEPS,
            "target_joints": {
                **joints,
                **{name: float(value) for name, value in zip(RING_JOINTS, values, strict=True)},
            },
            "planner": {
                "kind": "static ring-finger movable-face reachability sampling (not MPC, not learned)",
                "seed": SEED,
                "samples": SAMPLES,
            },
            "static_screen": {
                "side_geoms": side_geoms,
                "deepest_distance_m": deepest,
                "dynamic_pass": False,
            },
        }
        OUTPUT_POSE.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")

    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during reachability planning")
    lines = [
        "test=Prompt 24 static ring-finger movable-face side reachability",
        f"source_pose={SOURCE_POSE}",
        f"seed={SEED}",
        f"samples={SAMPLES}",
        f"target_contact_depth_m={TARGET_DEPTH_M:+.9f}",
        f"elapsed_s={elapsed:.6f}",
        f"safe_side_candidates={len(rows)}",
        f"rejected_penetration={rejected_penetration}",
        f"rejected_support={rejected_support}",
        f"rejected_ring_core_jam={rejected_core_jam}",
        f"best_joint_values={best[1].tolist() if best else 'none'}",
        f"best_side_geoms={best[2] if best else 'none'}",
        f"best_deepest_distance_m={best[3] if best else 'none'}",
        f"output_pose={OUTPUT_POSE if best else 'none'}",
        "dynamic_validation=NOT_RUN",
        "face_turn_attempted=NO",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
