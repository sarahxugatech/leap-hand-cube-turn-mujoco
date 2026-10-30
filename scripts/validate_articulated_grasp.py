"""Validate hold then lift for the TEST-only articulated face object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from validate_poses import (
    HAND_JOINTS,
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
TEST_SCENE = ROOT / "scenes" / "table_articulated_face_test.xml"
POSE = ROOT / "poses" / "articulated_grasp_R01.json"
LOG = ROOT / "logs" / "prompt23_articulated_grasp_R01.txt"
EXPECTED_MAIN_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
EXPECTED_TEST_SHA256 = "baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb"
HOLD_STEPS = 500
LIFT_RAMP_STEPS = 300
LIFT_HOLD_STEPS = 500
LIFT_M = 0.030
FACE_STABILITY_LIMIT = np.deg2rad(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=Path, default=POSE)
    parser.add_argument("--log", type=Path, default=LOG)
    args = parser.parse_args()
    pose = args.pose.resolve()
    log = args.log.resolve()
    main_before = sha256(MAIN_SCENE)
    test_before = sha256(TEST_SCENE)
    if main_before != EXPECTED_MAIN_SHA256:
        raise RuntimeError(f"frozen main scene hash mismatch: {main_before}")
    if test_before != EXPECTED_TEST_SHA256:
        raise RuntimeError(f"articulated TEST scene hash mismatch: {test_before}")

    payload = json.loads(pose.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(TEST_SCENE))
    if model.neq:
        raise RuntimeError("articulated TEST scene unexpectedly contains equality constraints")
    mount, joints, _, _ = load_pose(pose, model)
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

    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    object_geoms = {core_geom, face_geom}
    table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    hand = hand_geoms(model)
    hand_set = set(hand)
    finger_pairs = [
        (first, second)
        for index, first in enumerate(hand)
        for second in hand[index + 1 :]
        if finger_prefix(geom_name(model, first)) in {"if", "mf", "rf", "th"}
        and finger_prefix(geom_name(model, second)) in {"if", "mf", "rf", "th"}
        and finger_prefix(geom_name(model, first)) != finger_prefix(geom_name(model, second))
    ]

    minimum_hand_object = float("inf")
    minimum_hand_table = float("inf")
    minimum_finger_finger = float("inf")
    warnings: list[str] = []

    def observe() -> tuple[set[str], set[str], set[str]]:
        nonlocal minimum_hand_object, minimum_hand_table, minimum_finger_finger
        contacts: set[str] = set()
        core_contacts: set[str] = set()
        face_contacts: set[str] = set()
        for contact in data.contact[: data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            pair = {first, second}
            touched = pair & object_geoms
            if touched:
                object_geom = next(iter(touched))
                other = second if first == object_geom else first
                if other in hand_set:
                    name = geom_name(model, other)
                    contacts.add(name)
                    (face_contacts if object_geom == face_geom else core_contacts).add(name)
                    minimum_hand_object = min(minimum_hand_object, float(contact.dist))
            elif table_geom in pair:
                other = second if first == table_geom else first
                if other in hand_set:
                    minimum_hand_table = min(minimum_hand_table, float(contact.dist))
            elif first in hand_set and second in hand_set:
                a = finger_prefix(geom_name(model, first))
                b = finger_prefix(geom_name(model, second))
                if a in {"if", "mf", "rf", "th"} and b in {"if", "mf", "rf", "th"} and a != b:
                    minimum_finger_finger = min(minimum_finger_finger, float(contact.dist))
        minimum_hand_object = min(
            minimum_hand_object,
            *(float(mujoco.mj_geomDistance(model, data, obj, geom, 1.0, None)) for obj in object_geoms for geom in hand),
        )
        minimum_hand_table = min(
            minimum_hand_table,
            *(float(mujoco.mj_geomDistance(model, data, table_geom, geom, 1.0, None)) for geom in hand),
        )
        minimum_finger_finger = min(
            minimum_finger_finger,
            *(float(mujoco.mj_geomDistance(model, data, first, second, 0.01, None)) for first, second in finger_pairs),
        )
        return contacts, core_contacts, face_contacts

    hold_start = data.xpos[cube_body].copy()
    hold_face_start = float(data.qpos[face_qpos])
    hold_thumb: list[bool] = []
    hold_maximum_displacement = 0.0
    hold_maximum_face_drift = 0.0
    final_contacts: set[str] = set()
    final_core_contacts: set[str] = set()
    final_face_contacts: set[str] = set()
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(HOLD_STEPS + 1):
            contacts, core_contacts, face_contacts = observe()
            prefixes = {finger_prefix(name) for name in contacts}
            hold_thumb.append("th" in prefixes)
            hold_maximum_displacement = max(
                hold_maximum_displacement,
                float(np.linalg.norm(data.xpos[cube_body] - hold_start)),
            )
            hold_maximum_face_drift = max(
                hold_maximum_face_drift,
                abs(float(data.qpos[face_qpos]) - hold_face_start),
            )
            final_contacts, final_core_contacts, final_face_contacts = contacts, core_contacts, face_contacts
            if step < HOLD_STEPS:
                mujoco.mj_step(model, data)

        hold_end = data.xpos[cube_body].copy()
        hold_final_face = float(data.qpos[face_qpos])
        hold_prefixes = {finger_prefix(name) for name in final_contacts}
        penetration = min(minimum_hand_object, minimum_hand_table, minimum_finger_finger) < -PENETRATION_LIMIT
        hold_pass = (
            all(hold_thumb[-100:])
            and len(hold_prefixes & {"if", "mf", "rf"}) >= 2
            and hold_maximum_displacement < 0.015
            and hold_maximum_face_drift < FACE_STABILITY_LIMIT
            and not penetration
            and not warnings
        )

        lift_ran = hold_pass
        lift_start = data.xpos[cube_body].copy()
        lift_relative_start = lift_start - data.xpos[palm_body]
        lift_face_start = float(data.qpos[face_qpos])
        lift_maximum_relative_drift = 0.0
        lift_maximum_face_drift = 0.0
        lift_thumb: list[bool] = []
        if lift_ran:
            for step in range(LIFT_RAMP_STEPS + LIFT_HOLD_STEPS + 1):
                contacts, core_contacts, face_contacts = observe()
                prefixes = {finger_prefix(name) for name in contacts}
                lift_thumb.append("th" in prefixes)
                relative = data.xpos[cube_body] - data.xpos[palm_body]
                lift_maximum_relative_drift = max(
                    lift_maximum_relative_drift,
                    float(np.linalg.norm(relative - lift_relative_start)),
                )
                lift_maximum_face_drift = max(
                    lift_maximum_face_drift,
                    abs(float(data.qpos[face_qpos]) - lift_face_start),
                )
                final_contacts, final_core_contacts, final_face_contacts = contacts, core_contacts, face_contacts
                if step == LIFT_RAMP_STEPS + LIFT_HOLD_STEPS:
                    break
                phase = min((step + 1) / LIFT_RAMP_STEPS, 1.0)
                blend = 0.5 - 0.5 * np.cos(np.pi * phase)
                data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + LIFT_M * blend
                mujoco.mj_step(model, data)
    finally:
        mujoco.set_mju_user_warning(old_handler)

    warnings.extend(
        mujoco.mjtWarning(index).name
        for index in range(mujoco.mjtWarning.mjNWARNING)
        if data.warning[index].number
    )
    lift_end = data.xpos[cube_body].copy()
    lift_end_relative_drift = float(
        np.linalg.norm((lift_end - data.xpos[palm_body]) - lift_relative_start)
    )
    final_prefixes = {finger_prefix(name) for name in final_contacts}
    penetration = min(minimum_hand_object, minimum_hand_table, minimum_finger_finger) < -PENETRATION_LIMIT
    lift_pass = (
        lift_ran
        and float(lift_end[2] - lift_start[2]) >= 0.020
        and lift_maximum_relative_drift < 0.015
        and lift_end_relative_drift < 0.015
        and all(lift_thumb[-100:])
        and len(final_prefixes & {"if", "mf", "rf"}) >= 2
        and lift_maximum_face_drift < FACE_STABILITY_LIMIT
        and not penetration
        and not warnings
    )

    main_after = sha256(MAIN_SCENE)
    test_after = sha256(TEST_SCENE)
    if main_after != main_before or test_after != test_before:
        raise RuntimeError("scene changed during articulated grasp validation")
    lines = [
        "test=Prompt 23 articulated object hold then lift gate",
        f"pose={pose}",
        f"hold_steps={HOLD_STEPS}",
        f"lift_ramp_steps={LIFT_RAMP_STEPS}",
        f"lift_hold_steps={LIFT_HOLD_STEPS}",
        "cube_hand_attachment=NO",
        "cube_qpos_updates_during_run=0",
        f"hold_start_m={hold_start.tolist()}",
        f"hold_end_m={hold_end.tolist()}",
        f"hold_maximum_displacement_m={hold_maximum_displacement:.9f}",
        f"hold_end_displacement_m={np.linalg.norm(hold_end - hold_start):.9f}",
        f"hold_face_start_rad={hold_face_start:+.9f}",
        f"hold_face_end_rad={hold_final_face:+.9f}",
        f"hold_maximum_face_drift_rad={hold_maximum_face_drift:.9f}",
        f"hold_thumb_sustained={'YES' if all(hold_thumb[-100:]) else 'NO'}",
        f"HOLD_GATE={'PASS' if hold_pass else 'FAIL'}",
        f"lift_ran={'YES' if lift_ran else 'NO'}",
        f"lift_cube_rise_m={float(lift_end[2] - lift_start[2]):+.9f}",
        f"lift_maximum_palm_relative_drift_m={lift_maximum_relative_drift:.9f}",
        f"lift_end_palm_relative_drift_m={lift_end_relative_drift:.9f}",
        f"lift_maximum_face_drift_rad={lift_maximum_face_drift:.9f}",
        f"lift_thumb_sustained={'YES' if lift_thumb and all(lift_thumb[-100:]) else 'NO'}",
        f"final_fingers={sorted(final_prefixes & {'if', 'mf', 'rf'})}",
        f"final_core_contacts={sorted(final_core_contacts)}",
        f"final_face_layer_contacts={sorted(final_face_contacts)}",
        f"minimum_hand_object_distance_m={minimum_hand_object:+.9f}",
        f"minimum_finger_finger_distance_m={minimum_finger_finger:+.9f}",
        f"minimum_hand_table_distance_m={minimum_hand_table:+.9f}",
        f"penetration_over_0.002_m={'YES' if penetration else 'NO'}",
        f"warnings={warnings or 'none'}",
        f"LIFT_GATE={'PASS' if lift_pass else 'FAIL'}",
        "face_turn_attempted=NO",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={test_before}",
        f"test_scene_sha256_after={test_after}",
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
