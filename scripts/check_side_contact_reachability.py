"""Check whether index or middle distal geoms can safely reach a cube side at T01."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from validate_poses import (
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
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
    test_scene_copy,
)


CANDIDATE = ROOT / "poses" / "turn_ready_T01.json"
OUTPUT = ROOT / "logs" / "prompt15_side_contact_reachability.txt"
SAMPLES = 20_000
SIDE_MARGIN_M = 0.005
SEEDS = {"if": 15001, "mf": 15002}


def distal(name: str) -> bool:
    return "_md_" in name or "_ds_" in name or name.endswith("_tip")


def main() -> None:
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    source_pose = Path(candidate["source_held_pose"])
    source_payload = json.loads(source_pose.read_text(encoding="utf-8"))

    with TemporaryDirectory(prefix="prompt15_reach_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(MAIN_SCENE.read_text(encoding="utf-8"), Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, source_joints, _, _ = load_pose(source_pose, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        hand = hand_geoms(model)
        hand_set = set(hand)
        data.qpos[cube_qpos : cube_qpos + 7] = (
            *source_payload["cube_start_xyz"],
            *source_payload["cube_start_quat_wxyz"],
        )
        for name, value in mount.items() | source_joints.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        mujoco.mj_forward(model, data)
        for _ in range(500):
            mujoco.mj_step(model, data)
        for step in range(500):
            phase = min((step + 1) / 300, 1.0)
            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
            data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
            mujoco.mj_step(model, data)
        for step in range(int(candidate["transition_steps"])):
            phase = (step + 1) / int(candidate["transition_steps"])
            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
            for name, target in candidate["target_joints"].items():
                data.ctrl[actuator_id(model, name)] = source_joints[name] + blend * (target - source_joints[name])
            mujoco.mj_step(model, data)
        for _ in range(500):
            mujoco.mj_step(model, data)
        frozen_cube_qpos = data.qpos[cube_qpos : cube_qpos + 7].copy()
        frozen_other_qpos = data.qpos.copy()

        result_lines: list[str] = []
        for prefix in ("if", "mf"):
            rng = np.random.default_rng(SEEDS[prefix])
            names = tuple(f"{prefix}_{suffix}" for suffix in ("mcp", "rot", "pip", "dip"))
            joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
            lower = np.array([model.jnt_range[joint, 0] for joint in joint_ids])
            upper = np.array([model.jnt_range[joint, 1] for joint in joint_ids])
            distal_geoms = [
                geom for geom in hand
                if finger_prefix(geom_name(model, geom)) == prefix and distal(geom_name(model, geom))
            ]
            safe_side_count = 0
            best_safe_distance = float("inf")
            best_safe_values: np.ndarray | None = None
            safest_side_depth = -float("inf")
            safest_side_values: np.ndarray | None = None
            safest_side_geoms: list[str] = []
            for values in rng.uniform(lower, upper, size=(SAMPLES, 4)):
                data.qpos[:] = frozen_other_qpos
                data.qpos[cube_qpos : cube_qpos + 7] = frozen_cube_qpos
                for name, value in zip(names, values, strict=True):
                    data.qpos[qpos_address(model, name)] = value
                mujoco.mj_forward(model, data)
                deepest = 0.0
                side_geoms: list[str] = []
                cube_z = float(data.xpos[cube_body, 2])
                for contact in data.contact[: data.ncon]:
                    first = int(contact.geom1)
                    second = int(contact.geom2)
                    pair = {first, second}
                    if cube_geom in pair:
                        other = second if first == cube_geom else first
                        if other in hand_set:
                            deepest = min(deepest, float(contact.dist))
                            name = geom_name(model, other)
                            if (
                                finger_prefix(name) == prefix
                                and distal(name)
                                and float(contact.pos[2]) > cube_z - 0.025 + SIDE_MARGIN_M
                            ):
                                side_geoms.append(name)
                    elif first in hand_set and second in hand_set:
                        first_prefix = finger_prefix(geom_name(model, first))
                        second_prefix = finger_prefix(geom_name(model, second))
                        if first_prefix and second_prefix and first_prefix != second_prefix:
                            deepest = min(deepest, float(contact.dist))
                if deepest < -PENETRATION_LIMIT:
                    continue
                distance = min(
                    float(mujoco.mj_geomDistance(model, data, cube_geom, geom, 0.10, None))
                    for geom in distal_geoms
                )
                if distance < best_safe_distance:
                    best_safe_distance = distance
                    best_safe_values = values.copy()
                if side_geoms:
                    safe_side_count += 1
                    if deepest > safest_side_depth:
                        safest_side_depth = deepest
                        safest_side_values = values.copy()
                        safest_side_geoms = sorted(set(side_geoms))
            result_lines.extend((
                f"finger={prefix}",
                f"seed={SEEDS[prefix]}",
                f"samples={SAMPLES}",
                f"safe_distal_side_contacts={safe_side_count}",
                f"best_safe_distal_distance_m={best_safe_distance:+.9f}",
                f"best_safe_joint_values={best_safe_values.tolist() if best_safe_values is not None else 'none'}",
                f"safest_side_contact_depth_m={safest_side_depth:+.9f}",
                f"safest_side_joint_values={safest_side_values.tolist() if safest_side_values is not None else 'none'}",
                f"safest_side_geoms={safest_side_geoms or 'none'}",
                "",
            ))

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during reachability check")
    lines = [
        "test=Prompt 15 static distal-side reachability gate at validated T01 state",
        f"candidate={CANDIDATE}",
        f"side_margin_m={SIDE_MARGIN_M}",
        "cube_state=frozen after real R01 hold, lift, T01 transition, and 500-step hold",
        "dynamics_during_sampling=NO",
        "rotation_run=NO",
        *result_lines,
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
