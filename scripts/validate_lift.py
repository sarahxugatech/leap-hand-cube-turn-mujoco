"""Validate a TEST-only LEAP lift after a held pose passed the 500-step gate."""

from __future__ import annotations

import argparse
import json
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


RAMP_STEPS = 300
HOLD_STEPS = 500
LIFT_M = 0.030


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=Path, required=True)
    args = parser.parse_args()
    pose = args.pose.resolve()
    log = ROOT / "logs" / f"lift_{pose.stem}.txt"
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(pose.read_text(encoding="utf-8"))
    if payload.get("test_only") is not True:
        raise ValueError("lift pose must be marked test_only")

    source = MAIN_SCENE.read_text(encoding="utf-8")
    with TemporaryDirectory(prefix="lift_gate_", dir=ROOT / "logs") as temporary:
        yaw_enabled = "mount_yaw_degrees" in payload
        scene = test_scene_copy(
            source,
            Path(temporary),
            enable_yaw=yaw_enabled,
            mount_quat=payload.get("mount_quat_wxyz"),
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, joints, yaw_degrees, _ = load_pose(pose, model)
        if model.neq != 0:
            raise RuntimeError("TEST scene unexpectedly contains equality constraints")
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        data.qpos[cube_qpos : cube_qpos + 7] = (
            *payload["cube_start_xyz"],
            *payload["cube_start_quat_wxyz"],
        )
        for name, value in mount.items() | joints.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        if yaw_enabled:
            yaw = float(np.deg2rad(yaw_degrees))
            data.qpos[qpos_address(model, "mount_yaw")] = yaw
            data.ctrl[actuator_id(model, "mount_yaw")] = yaw
        mujoco.mj_forward(model, data)

        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
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
        cube_start = data.xpos[cube_body].copy()
        relative_start = cube_start - data.xpos[palm_body]
        target_mount_z = mount["mount_z"] + LIFT_M
        thumb_history: list[bool] = []
        final_fingers: set[str] = set()
        minimum_hand_cube = float("inf")
        minimum_hand_table = float("inf")
        minimum_finger_finger = float("inf")
        maximum_relative_drift = 0.0
        warnings: list[str] = []
        old_handler = mujoco.set_mju_user_warning(warnings.append)
        try:
            for step in range(RAMP_STEPS + HOLD_STEPS + 1):
                contacts: list[str] = []
                for contact in data.contact[: data.ncon]:
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if cube_geom in pair:
                        other = int(contact.geom2 if contact.geom1 == cube_geom else contact.geom1)
                        if other in hand_set:
                            contacts.append(geom_name(model, other))
                prefixes = {finger_prefix(name) for name in contacts}
                thumb_history.append("th" in prefixes)
                final_fingers = prefixes & {"if", "mf", "rf"}
                minimum_hand_cube = min(
                    minimum_hand_cube,
                    *(float(mujoco.mj_geomDistance(model, data, cube_geom, geom, 1.0, None)) for geom in hand),
                )
                minimum_hand_table = min(
                    minimum_hand_table,
                    *(float(mujoco.mj_geomDistance(model, data, table_geom, geom, 1.0, None)) for geom in hand),
                )
                minimum_finger_finger = min(
                    minimum_finger_finger,
                    *(float(mujoco.mj_geomDistance(model, data, first, second, 0.01, None)) for first, second in finger_pairs),
                )
                relative = data.xpos[cube_body] - data.xpos[palm_body]
                maximum_relative_drift = max(maximum_relative_drift, float(np.linalg.norm(relative - relative_start)))
                if step == RAMP_STEPS + HOLD_STEPS:
                    break
                phase = min((step + 1) / RAMP_STEPS, 1.0)
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
        cube_end = data.xpos[cube_body].copy()
        cube_rise = float(cube_end[2] - cube_start[2])
        end_relative_drift = float(
            np.linalg.norm((cube_end - data.xpos[palm_body]) - relative_start)
        )
        thumb_sustained = all(thumb_history[-100:])
        penetration = min(minimum_hand_cube, minimum_hand_table, minimum_finger_finger) < -PENETRATION_LIMIT
        passed = (
            cube_rise >= 0.020
            and maximum_relative_drift < 0.015
            and end_relative_drift < 0.015
            and thumb_sustained
            and len(final_fingers) >= 2
            and not penetration
            and not warnings
        )

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during lift gate")
    lines = [
        "test=TEST-scene lift gate",
        f"pose={pose}",
        f"ramp_steps={RAMP_STEPS}",
        f"hold_steps={HOLD_STEPS}",
        f"mount_yaw_degrees={yaw_degrees:.9f}",
        f"commanded_lift_m={LIFT_M:.6f}",
        "cube_hand_attachment=NO",
        "cube_qpos_updates_during_run=0",
        f"cube_start_m={np.array2string(cube_start, precision=9, separator=', ')}",
        f"cube_end_m={np.array2string(cube_end, precision=9, separator=', ')}",
        f"cube_rise_m={cube_rise:+.9f}",
        f"maximum_palm_relative_drift_m={maximum_relative_drift:.9f}",
        f"end_palm_relative_drift_m={end_relative_drift:.9f}",
        f"thumb_sustained_last_100={'YES' if thumb_sustained else 'NO'}",
        f"final_fingers={sorted(final_fingers)}",
        f"minimum_hand_cube_distance_m={minimum_hand_cube:+.9f}",
        f"minimum_finger_finger_distance_m={minimum_finger_finger:+.9f}",
        f"minimum_hand_table_distance_m={minimum_hand_table:+.9f}",
        f"penetration_over_0.002_m={'YES' if penetration else 'NO'}",
        f"warnings={warnings or 'none'}",
        f"LIFT_GATE={'PASS' if passed else 'FAIL'}",
        "rotation_run=NO",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
