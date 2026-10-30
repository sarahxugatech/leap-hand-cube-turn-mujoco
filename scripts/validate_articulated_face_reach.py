"""Dynamically validate the ring-finger movable-face reach candidate."""

from __future__ import annotations

import json
import math
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
CANDIDATE = ROOT / "poses" / "articulated_face_reach_RF02.json"
OUTPUT = ROOT / "logs" / "prompt24_articulated_face_reach_RF02_validation.txt"
EXPECTED_MAIN_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
EXPECTED_SCENE_SHA256 = "baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb"
RING_JOINTS = ("rf_mcp", "rf_rot", "rf_pip", "rf_dip")
HOLD_STEPS = 500
FACE_DRIFT_LIMIT = math.radians(2.0)


def distal(name: str) -> bool:
    return "_md_" in name or "_ds_" in name or name.endswith("_tip")


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before reach validation")
    payload = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
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

    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    hand = hand_geoms(model)
    hand_set = set(hand)
    start_relative = data.xpos[cube_body] - data.xpos[palm_body]
    start_face = float(data.qpos[face_qpos])
    transition_steps = int(candidate["transition_steps"])
    target = candidate["target_joints"]
    support_history: list[bool] = []
    side_history: list[bool] = []
    ring_core_history: list[bool] = []
    maximum_relative_drift = 0.0
    maximum_face_drift = 0.0
    minimum_hand_object = float("inf")
    minimum_hand_table = float("inf")
    minimum_finger_finger = float("inf")
    final_side_geoms: set[str] = set()
    warnings: list[str] = []

    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(transition_steps + HOLD_STEPS):
            if step < transition_steps:
                phase = (step + 1) / transition_steps
                blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                for name in RING_JOINTS:
                    value = joints[name] + blend * (float(target[name]) - joints[name])
                    data.ctrl[actuator_id(model, name)] = value
            mujoco.mj_step(model, data)
            core_prefixes: set[str] = set()
            side_geoms: set[str] = set()
            ring_core = False
            rotation = data.xmat[face_body].reshape(3, 3)
            center = data.xpos[face_body]
            for contact in data.contact[: data.ncon]:
                first, second = int(contact.geom1), int(contact.geom2)
                pair = {first, second}
                if core_geom in pair:
                    other = second if first == core_geom else first
                    if other in hand_set:
                        minimum_hand_object = min(minimum_hand_object, float(contact.dist))
                        prefix = finger_prefix(geom_name(model, other))
                        if prefix:
                            core_prefixes.add(prefix)
                        ring_core |= prefix == "rf"
                elif face_geom in pair:
                    other = second if first == face_geom else first
                    if other in hand_set:
                        minimum_hand_object = min(minimum_hand_object, float(contact.dist))
                        name = geom_name(model, other)
                        local = rotation.T @ (np.asarray(contact.pos) - center)
                        on_side = max(abs(local[0]), abs(local[1])) >= 0.023 and abs(local[2]) <= 0.0085
                        if finger_prefix(name) == "rf" and distal(name) and on_side:
                            side_geoms.add(name)
                elif table_geom in pair:
                    other = second if first == table_geom else first
                    if other in hand_set:
                        minimum_hand_table = min(minimum_hand_table, float(contact.dist))
                elif first in hand_set and second in hand_set:
                    a = finger_prefix(geom_name(model, first))
                    b = finger_prefix(geom_name(model, second))
                    if a and b and a != b:
                        minimum_finger_finger = min(minimum_finger_finger, float(contact.dist))
            support_history.append({"th", "if", "mf"}.issubset(core_prefixes))
            side_history.append(bool(side_geoms))
            ring_core_history.append(ring_core)
            final_side_geoms = side_geoms
            relative = data.xpos[cube_body] - data.xpos[palm_body]
            maximum_relative_drift = max(
                maximum_relative_drift,
                float(np.linalg.norm(relative - start_relative)),
            )
            maximum_face_drift = max(
                maximum_face_drift,
                abs(float(data.qpos[face_qpos]) - start_face),
            )
    finally:
        mujoco.set_mju_user_warning(old_handler)

    warnings.extend(
        mujoco.mjtWarning(index).name
        for index in range(mujoco.mjtWarning.mjNWARNING)
        if data.warning[index].number
    )
    penetration = min(minimum_hand_object, minimum_hand_table, minimum_finger_finger) < -PENETRATION_LIMIT
    side_steps = [index for index, value in enumerate(side_history) if value]
    side_fraction_last_100 = sum(side_history[-100:]) / 100
    passed = (
        all(support_history[-100:])
        and all(side_history[-100:])
        and not any(ring_core_history[-100:])
        and maximum_relative_drift < 0.015
        and maximum_face_drift < FACE_DRIFT_LIMIT
        and not penetration
        and not warnings
    )
    candidate["static_screen"]["dynamic_pass"] = passed
    if passed:
        CANDIDATE.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")

    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during reach validation")
    lines = [
        "test=Prompt 24 dynamic articulated-face reach validation",
        f"candidate={CANDIDATE}",
        f"transition_steps={transition_steps}",
        f"hold_steps={HOLD_STEPS}",
        f"support_sustained_last_100={'YES' if all(support_history[-100:]) else 'NO'}",
        f"face_side_contact_sustained_last_100={'YES' if all(side_history[-100:]) else 'NO'}",
        f"face_side_contact_fraction_last_100={side_fraction_last_100:.6f}",
        f"first_face_side_contact_step={side_steps[0] if side_steps else 'none'}",
        f"last_face_side_contact_step={side_steps[-1] if side_steps else 'none'}",
        f"ring_core_contact_last_100={'YES' if any(ring_core_history[-100:]) else 'NO'}",
        f"final_side_geoms={sorted(final_side_geoms)}",
        f"maximum_palm_relative_drift_m={maximum_relative_drift:.9f}",
        f"face_end_angle_rad={float(data.qpos[face_qpos]):+.9f}",
        f"maximum_face_drift_rad={maximum_face_drift:.9f}",
        f"minimum_hand_object_distance_m={minimum_hand_object:+.9f}",
        f"minimum_finger_finger_distance_m={minimum_finger_finger:+.9f}",
        f"minimum_hand_table_distance_m={minimum_hand_table:+.9f}",
        f"penetration_over_0.002_m={'YES' if penetration else 'NO'}",
        f"warnings={warnings or 'none'}",
        f"FACE_REACH_GATE={'PASS' if passed else 'FAIL'}",
        "face_turn_attempted=NO",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
