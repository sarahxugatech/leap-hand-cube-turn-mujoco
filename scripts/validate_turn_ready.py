"""Independently validate a planned turn-ready grasp transition for 500 steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
import imageio.v2 as imageio

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
    sha256,
    test_scene_copy,
)


HOLD_STEPS = 500
SUSTAINED_STEPS = 100
SIDE_MARGIN_M = 0.005


def distal(name: str) -> bool:
    return "_md_" in name or "_ds_" in name or name.endswith("_tip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    source_pose = Path(payload["source_held_pose"])
    target_joints = {name: float(payload["target_joints"][name]) for name in HAND_JOINTS}
    transition_steps = int(payload["transition_steps"])
    lift_m = float(payload["mount_lift_m"])
    log = ROOT / "logs" / f"validate_{candidate.stem}.txt"

    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    source_payload = json.loads(source_pose.read_text(encoding="utf-8"))

    with TemporaryDirectory(prefix="turn_ready_validate_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(MAIN_SCENE.read_text(encoding="utf-8"), Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, source_joints, _, _ = load_pose(source_pose, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        hand = hand_geoms(model)
        hand_set = set(hand)
        for name, value in target_joints.items():
            joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lower, upper = model.jnt_range[joint]
            if not lower <= value <= upper:
                raise ValueError(f"{name}={value} outside [{lower}, {upper}]")

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
            data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + lift_m * blend
            mujoco.mj_step(model, data)

        transition_start = data.xpos[cube_body].copy()
        relative_start = transition_start - data.xpos[palm_body]
        transition_start_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
        thumb_history: list[bool] = []
        support_history: list[bool] = []
        side_history: list[bool] = []
        final_contacts: set[str] = set()
        final_side_contacts: set[str] = set()
        minimum_hand_cube = float("inf")
        minimum_hand_table = float("inf")
        minimum_finger_finger = float("inf")
        minimum_cube_z = float("inf")
        maximum_displacement = 0.0
        maximum_relative_drift = 0.0
        warnings: list[str] = []
        snapshots: list[np.ndarray] = []
        capture_every = max(1, round((1.0 / 30.0) / model.opt.timestep))

        old_handler = mujoco.set_mju_user_warning(warnings.append)
        try:
            for step in range(transition_steps + HOLD_STEPS + 1):
                if 0 < step <= transition_steps:
                    phase = step / transition_steps
                    blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                    for name, target_value in target_joints.items():
                        start_value = source_joints[name]
                        data.ctrl[actuator_id(model, name)] = start_value + blend * (target_value - start_value)
                    mujoco.mj_step(model, data)
                elif step > transition_steps:
                    mujoco.mj_step(model, data)

                contacts: set[str] = set()
                side_contacts: set[str] = set()
                cube_z = float(data.xpos[cube_body, 2])
                for contact in data.contact[: data.ncon]:
                    first = int(contact.geom1)
                    second = int(contact.geom2)
                    pair = {first, second}
                    if cube_geom in pair:
                        other = second if first == cube_geom else first
                        if other in hand_set:
                            name = geom_name(model, other)
                            contacts.add(name)
                            minimum_hand_cube = min(minimum_hand_cube, float(contact.dist))
                            if (
                                finger_prefix(name) in {"if", "mf", "rf"}
                                and distal(name)
                                and float(contact.pos[2]) > cube_z - 0.025 + SIDE_MARGIN_M
                            ):
                                side_contacts.add(name)
                    elif table_geom in pair:
                        other = second if first == table_geom else first
                        if other in hand_set:
                            minimum_hand_table = min(minimum_hand_table, float(contact.dist))
                    elif first in hand_set and second in hand_set:
                        first_prefix = finger_prefix(geom_name(model, first))
                        second_prefix = finger_prefix(geom_name(model, second))
                        if (
                            first_prefix in {"if", "mf", "rf", "th"}
                            and second_prefix in {"if", "mf", "rf", "th"}
                            and first_prefix != second_prefix
                        ):
                            minimum_finger_finger = min(minimum_finger_finger, float(contact.dist))
                prefixes = {finger_prefix(name) for name in contacts}
                thumb_history.append("th" in prefixes)
                support_history.append(len(prefixes & {"if", "mf", "rf"}) >= 2)
                side_history.append(bool(side_contacts))
                final_contacts = contacts
                final_side_contacts = side_contacts
                position = data.xpos[cube_body]
                relative = position - data.xpos[palm_body]
                maximum_displacement = max(
                    maximum_displacement, float(np.linalg.norm(position - transition_start))
                )
                maximum_relative_drift = max(
                    maximum_relative_drift, float(np.linalg.norm(relative - relative_start))
                )
                minimum_cube_z = min(minimum_cube_z, float(position[2]))
                if step % capture_every == 0:
                    snapshots.append(data.qpos.copy())
        finally:
            mujoco.set_mju_user_warning(old_handler)

        warnings.extend(
            mujoco.mjtWarning(index).name
            for index in range(mujoco.mjtWarning.mjNWARNING)
            if data.warning[index].number
        )
        final_prefixes = {finger_prefix(name) for name in final_contacts}
        final_fingers = sorted(final_prefixes & {"if", "mf", "rf"})
        thumb_sustained = all(thumb_history[-SUSTAINED_STEPS:])
        supports_sustained = all(support_history[-SUSTAINED_STEPS:])
        side_sustained = all(side_history[-SUSTAINED_STEPS:])
        penetration = min(minimum_hand_cube, minimum_hand_table, minimum_finger_finger) < -PENETRATION_LIMIT
        drop = minimum_cube_z < 0.105
        end_displacement = float(np.linalg.norm(data.xpos[cube_body] - transition_start))
        end_relative_drift = float(
            np.linalg.norm((data.xpos[cube_body] - data.xpos[palm_body]) - relative_start)
        )
        final_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
        passed = (
            thumb_sustained
            and supports_sustained
            and side_sustained
            and len(final_fingers) >= 2
            and maximum_relative_drift < 0.015
            and end_relative_drift < 0.015
            and not penetration
            and not drop
            and not warnings
        )
        video = ROOT / "videos" / f"prompt14_{candidate.stem}_transition.mp4"
        video_hash = "none"
        if args.video:
            video.parent.mkdir(exist_ok=True)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = source_payload["cube_start_xyz"]
            camera.lookat[2] += lift_m
            camera.distance = 0.34
            camera.azimuth = 135
            camera.elevation = -18
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                with imageio.get_writer(video, fps=30, codec="libx264", quality=8) as writer:
                    for qpos in snapshots:
                        data.qpos[:] = qpos
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera)
                        writer.append_data(renderer.render())
            video_hash = hashlib.sha256(video.read_bytes()).hexdigest()

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during Prompt 14 validation")
    lines = [
        "test=Prompt 14 independent turn-ready transition validation",
        f"candidate={candidate}",
        f"source_pose={source_pose}",
        f"transition_steps={transition_steps}",
        f"hold_steps={HOLD_STEPS}",
        f"side_margin_m={SIDE_MARGIN_M}",
        "cube_hand_attachment=NO",
        "cube_qpos_updates_during_run=0",
        f"transition_start_quat_wxyz={transition_start_quat.tolist()}",
        f"final_quat_wxyz={final_quat.tolist()}",
        f"maximum_displacement_m={maximum_displacement:.9f}",
        f"end_displacement_m={end_displacement:.9f}",
        f"maximum_palm_relative_drift_m={maximum_relative_drift:.9f}",
        f"end_palm_relative_drift_m={end_relative_drift:.9f}",
        f"thumb_sustained_last_100={'YES' if thumb_sustained else 'NO'}",
        f"supports_sustained_last_100={'YES' if supports_sustained else 'NO'}",
        f"side_distal_sustained_last_100={'YES' if side_sustained else 'NO'}",
        f"final_fingers={final_fingers}",
        f"final_contacts={sorted(final_contacts)}",
        f"final_side_contacts={sorted(final_side_contacts)}",
        f"minimum_cube_center_z_m={minimum_cube_z:.9f}",
        f"minimum_hand_cube_distance_m={minimum_hand_cube:+.9f}",
        f"minimum_finger_finger_distance_m={minimum_finger_finger:+.9f}",
        f"minimum_hand_table_distance_m={minimum_hand_table:+.9f}",
        f"drop={'YES' if drop else 'NO'}",
        f"penetration_over_0.002_m={'YES' if penetration else 'NO'}",
        f"warnings={warnings or 'none'}",
        f"TURN_READY_GATE={'PASS' if passed else 'FAIL'}",
        "rotation_run=NO",
        f"video={video if args.video else 'none'}",
        f"video_sha256={video_hash}",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
