"""Validate a cache-initialized suspended cube for 500 steps without a cube attachment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from validate_poses import (
    EXPECTED_SCENE_SHA256,
    HAND_JOINTS,
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
    segment,
    sha256,
    test_scene_copy,
)


DEFAULT_POSE = ROOT / "poses" / "mjlab_left5_held_test.json"
STEPS = 500
SUSTAINED_STEPS = 100


def yes(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=Path, default=DEFAULT_POSE)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    pose = args.pose.resolve()
    log = args.log.resolve() if args.log else ROOT / "logs" / f"hold_{pose.stem}.txt"
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(pose.read_text(encoding="utf-8"))
    if payload.get("test_only") is not True:
        raise ValueError("held pose must be marked test_only")
    cube_start = np.asarray(payload["cube_start_xyz"], dtype=float)
    cube_quat = np.asarray(payload["cube_start_quat_wxyz"], dtype=float)

    source = MAIN_SCENE.read_text(encoding="utf-8")
    with TemporaryDirectory(prefix="held_pose_", dir=ROOT / "logs") as temporary:
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
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        if model.jnt_type[cube_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("cube is not a free body")

        data = mujoco.MjData(model)
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        data.qpos[cube_qpos : cube_qpos + 7] = (*cube_start, *cube_quat)
        for name, value in mount.items() | joints.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        if yaw_enabled:
            yaw = float(np.deg2rad(yaw_degrees))
            data.qpos[qpos_address(model, "mount_yaw")] = yaw
            data.ctrl[actuator_id(model, "mount_yaw")] = yaw
        mujoco.mj_forward(model, data)

        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
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

        minimum_hand_cube = float("inf")
        minimum_hand_table = float("inf")
        minimum_finger_finger = float("inf")
        maximum_displacement = 0.0
        thumb_history: list[bool] = []
        initial_contacts: list[str] = []
        final_contacts: list[str] = []
        last_contact_step: int | None = None
        last_thumb_step: int | None = None
        warnings: list[str] = []
        old_handler = mujoco.set_mju_user_warning(warnings.append)
        try:
            for step in range(STEPS + 1):
                position = data.xpos[cube_body].copy()
                maximum_displacement = max(
                    maximum_displacement, float(np.linalg.norm(position - cube_start))
                )
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
                contacts = []
                for contact in data.contact[: data.ncon]:
                    geoms = {int(contact.geom1), int(contact.geom2)}
                    if cube_geom not in geoms:
                        continue
                    other = int(contact.geom2 if contact.geom1 == cube_geom else contact.geom1)
                    if other in hand_set:
                        contacts.append(geom_name(model, other))
                thumb_history.append(any(finger_prefix(name) == "th" for name in contacts))
                if step == 0:
                    initial_contacts = contacts.copy()
                if contacts:
                    last_contact_step = step
                if thumb_history[-1]:
                    last_thumb_step = step
                if step == STEPS:
                    final_contacts = contacts
                    break
                mujoco.mj_step(model, data)
        finally:
            mujoco.set_mju_user_warning(old_handler)

        warnings.extend(
            mujoco.mjtWarning(index).name
            for index in range(mujoco.mjtWarning.mjNWARNING)
            if data.warning[index].number
        )
        cube_end = data.xpos[cube_body].copy()
        end_displacement = float(np.linalg.norm(cube_end - cube_start))
        z_change = float(cube_end[2] - cube_start[2])
        final_fingers = sorted({
            prefix for name in final_contacts
            if (prefix := finger_prefix(name)) in {"if", "mf", "rf"}
        })
        tip_fingers = sorted({
            prefix for name in final_contacts
            if segment(name) == "tip" and (prefix := finger_prefix(name)) in {"if", "mf", "rf"}
        })
        thumb_sustained = all(thumb_history[-SUSTAINED_STEPS:])
        penetration = min(minimum_hand_cube, minimum_hand_table, minimum_finger_finger) < -PENETRATION_LIMIT
        safe = not penetration and not warnings
        gate_a = thumb_sustained and len(tip_fingers) >= 2 and maximum_displacement < 0.005 and end_displacement < 0.005 and safe
        relaxed = thumb_sustained and len(final_fingers) >= 2 and maximum_displacement < 0.015 and end_displacement < 0.015 and safe
        stability = relaxed and cube_end[2] - 0.025 > 0.05

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during held-pose validation")
    lines = [
        "test=TEST-scene initialized held pose passive hold",
        f"pose={pose}",
        f"steps={STEPS}",
        f"mount_yaw_degrees={yaw_degrees:.9f}",
        "cube_hand_attachment=NO",
        "cube_freejoint=YES",
        "equality_constraints=0",
        "cube_qpos_updates_during_run=0",
        f"cube_start_m={np.array2string(cube_start, precision=9, separator=', ')}",
        f"cube_end_m={np.array2string(cube_end, precision=9, separator=', ')}",
        f"maximum_3d_displacement_m={maximum_displacement:.9f}",
        f"end_3d_displacement_m={end_displacement:.9f}",
        f"z_change_m={z_change:+.9f}",
        f"thumb_sustained={yes(thumb_sustained)}",
        f"initial_contacts={initial_contacts or 'none'}",
        f"last_hand_contact_step={last_contact_step if last_contact_step is not None else 'none'}",
        f"last_thumb_contact_step={last_thumb_step if last_thumb_step is not None else 'none'}",
        f"final_fingers={final_fingers}",
        f"tip_fingers={tip_fingers}",
        f"final_contacts={final_contacts or 'none'}",
        f"minimum_hand_cube_distance_m={minimum_hand_cube:+.9f}",
        f"minimum_finger_finger_distance_m={minimum_finger_finger:+.9f}",
        f"minimum_hand_table_distance_m={minimum_hand_table:+.9f}",
        f"penetration_over_0.002_m={yes(penetration)}",
        f"warnings={warnings or 'none'}",
        f"GATE_A={'PASS' if gate_a else 'FAIL'}",
        f"RELAXED={'PASS' if relaxed else 'FAIL'}",
        f"PASSIVE_HOLD_STABILITY={'PASS' if stability else 'FAIL'}",
        "main_scene_updated=NO",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
