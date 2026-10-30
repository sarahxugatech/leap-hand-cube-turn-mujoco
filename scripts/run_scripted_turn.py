"""Run the first deterministic LEAP scripted Z-turn diagnostic from R01."""

from __future__ import annotations

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


POSE = ROOT / "poses" / "grasp_hold_refined_R01.json"
KEYFRAMES = ROOT / "poses" / "turn_keyframes_K01.json"
LOG = ROOT / "logs" / "prompt13_scripted_turn_run06_K01_2cycles.txt"
VIDEO = ROOT / "videos" / "prompt13_K01_two_cycle_failure.mp4"
CYCLES = 2
TARGET_DEG = 90.0
TOLERANCE_DEG = 10.0


class SafetyAbort(RuntimeError):
    """Stop a trajectory at its first hard safety-gate violation."""


def quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0])


def quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def relative_angles_degrees(start: np.ndarray, end: np.ndarray) -> tuple[float, float, float]:
    relative = quat_multiply(end, quat_conjugate(start))
    relative /= np.linalg.norm(relative)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, relative)
    yaw = math.degrees(math.atan2(matrix[3], matrix[0]))
    tilt = math.degrees(math.acos(float(np.clip(matrix[8], -1.0, 1.0))))
    geodesic = math.degrees(2.0 * math.acos(float(np.clip(abs(relative[0]), 0.0, 1.0))))
    return yaw, tilt, geodesic


def main() -> None:
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(POSE.read_text(encoding="utf-8"))
    keyframes = json.loads(KEYFRAMES.read_text(encoding="utf-8"))["keyframes"]
    source = MAIN_SCENE.read_text(encoding="utf-8")

    with TemporaryDirectory(prefix="prompt13_turn_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(source, Path(temporary), mount_quat=payload.get("mount_quat_wxyz"))
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, joints, _, _ = load_pose(POSE, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        hand = hand_geoms(model)
        hand_set = set(hand)
        data.qpos[cube_qpos : cube_qpos + 7] = (
            *payload["cube_start_xyz"],
            *payload["cube_start_quat_wxyz"],
        )
        command = mount | joints
        for name, value in command.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        mujoco.mj_forward(model, data)

        minimum_hand_cube = float("inf")
        minimum_hand_table = float("inf")
        minimum_finger_finger = float("inf")
        minimum_cube_z = float("inf")
        maximum_palm_relative_drift = 0.0
        phase_rows: list[str] = []
        thumb_history: list[bool] = []
        contact_history: list[set[str]] = []
        warnings: list[str] = []
        initial_relative = data.xpos[cube_body] - data.xpos[palm_body]
        step_count = 0
        turn_start_quat: np.ndarray | None = None
        abort_reason: str | None = None
        snapshots: list[np.ndarray] = []
        capture_every = max(1, round((1.0 / 30.0) / model.opt.timestep))

        def observe() -> set[str]:
            nonlocal minimum_hand_cube, minimum_hand_table, minimum_finger_finger
            nonlocal minimum_cube_z, maximum_palm_relative_drift
            contacts: set[str] = set()
            for contact in data.contact[: data.ncon]:
                first = int(contact.geom1)
                second = int(contact.geom2)
                pair = {first, second}
                if cube_geom in pair:
                    other = second if first == cube_geom else first
                    if other in hand_set:
                        contacts.add(geom_name(model, other))
                        minimum_hand_cube = min(minimum_hand_cube, float(contact.dist))
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
            contact_history.append(prefixes)
            minimum_cube_z = min(minimum_cube_z, float(data.xpos[cube_body, 2]))
            relative = data.xpos[cube_body] - data.xpos[palm_body]
            maximum_palm_relative_drift = max(
                maximum_palm_relative_drift,
                float(np.linalg.norm(relative - initial_relative)),
            )
            deepest = min(minimum_hand_cube, minimum_hand_table, minimum_finger_finger)
            if deepest < -PENETRATION_LIMIT:
                raise SafetyAbort(f"penetration at step {step_count}: {deepest:+.9f} m")
            if maximum_palm_relative_drift >= 0.015:
                raise SafetyAbort(
                    f"palm-relative drift at step {step_count}: "
                    f"{maximum_palm_relative_drift:.9f} m"
                )
            if minimum_cube_z < 0.105:
                raise SafetyAbort(f"drop at step {step_count}: cube Z {minimum_cube_z:.9f} m")
            if step_count % capture_every == 0:
                snapshots.append(data.qpos.copy())
            return contacts

        def run_phase(name: str, target: dict[str, float], steps: int) -> None:
            nonlocal command, step_count
            start = command.copy()
            for local_step in range(steps):
                phase = (local_step + 1) / steps
                blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                for joint_name, target_value in target.items():
                    value = start[joint_name] + blend * (target_value - start[joint_name])
                    data.ctrl[actuator_id(model, joint_name)] = value
                    command[joint_name] = value
                mujoco.mj_step(model, data)
                step_count += 1
                contacts = observe()
            prefixes = sorted({finger_prefix(value) for value in contacts if finger_prefix(value)})
            phase_rows.append(
                f"{name} | {steps} | {prefixes} | "
                f"cube_xyz={data.xpos[cube_body].tolist()} | quat={data.qpos[cube_qpos + 3:cube_qpos + 7].tolist()}"
            )

        old_handler = mujoco.set_mju_user_warning(warnings.append)
        try:
            observe()
            run_phase("start_hold", {}, 500)
            run_phase("lift", {"mount_z": mount["mount_z"] + 0.030}, 300)
            run_phase("lift_hold", {}, 200)
            turn_start_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
            run_phase(
                "ring_guard",
                keyframes["ring_guard"],
                300,
            )
            for cycle in range(1, CYCLES + 1):
                run_phase(
                    f"cycle_{cycle:02d}_differential_push",
                    keyframes["differential_push"],
                    300,
                )
                run_phase(
                    f"cycle_{cycle:02d}_release",
                    keyframes["release"],
                    100,
                )
                run_phase(
                    f"cycle_{cycle:02d}_reset",
                    {"if_rot": joints["if_rot"], "mf_rot": joints["mf_rot"]},
                    180,
                )
                run_phase(
                    f"cycle_{cycle:02d}_reclose",
                    {"if_mcp": joints["if_mcp"], "mf_mcp": joints["mf_mcp"]},
                    140,
                )
            run_phase("end_hold", {}, 500)
        except SafetyAbort as error:
            abort_reason = str(error)
        finally:
            mujoco.set_mju_user_warning(old_handler)

        warnings.extend(
            mujoco.mjtWarning(index).name
            for index in range(mujoco.mjtWarning.mjNWARNING)
            if data.warning[index].number
        )
        assert turn_start_quat is not None
        turn_end_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
        yaw_deg, tilt_deg, geodesic_deg = relative_angles_degrees(turn_start_quat, turn_end_quat)
        final_prefixes = contact_history[-1]
        thumb_sustained = all("th" in prefixes for prefixes in contact_history[-100:])
        final_fingers = sorted(final_prefixes & {"if", "mf", "rf"})
        penetration = min(minimum_hand_cube, minimum_hand_table, minimum_finger_finger) < -PENETRATION_LIMIT
        drop = minimum_cube_z < 0.105
        rotation_gate = abs(abs(yaw_deg) - TARGET_DEG) <= TOLERANCE_DEG and tilt_deg <= TOLERANCE_DEG
        end_hold_gate = abort_reason is None and thumb_sustained and len(final_fingers) >= 2
        passed = rotation_gate and end_hold_gate and not penetration and not drop and not warnings

        VIDEO.parent.mkdir(exist_ok=True)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = payload["cube_start_xyz"]
        camera.lookat[2] += 0.015
        camera.distance = 0.34
        camera.azimuth = 135
        camera.elevation = -18
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            with imageio.get_writer(VIDEO, fps=30, codec="libx264", quality=8) as writer:
                for qpos in snapshots:
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera)
                    writer.append_data(renderer.render())
        video_hash = hashlib.sha256(VIDEO.read_bytes()).hexdigest()

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during scripted turn")
    lines = [
        "test=Prompt 13 sampled-keyframe actor two-cycle boundary run 06 K01",
        f"pose={POSE}",
        f"keyframes={KEYFRAMES}",
        f"gait_cycles={CYCLES}",
        f"target_deg={TARGET_DEG:.1f}",
        f"tolerance_deg={TOLERANCE_DEG:.1f}",
        "controller=deterministic actor using saved keyframes",
        "sampled_planner=YES (offline; actor execution is separate)",
        "learned_policy=NO",
        "MPC=NO",
        "cube_hand_attachment=NO",
        "cube_qpos_updates_during_run=0",
        f"steps={step_count}",
        f"turn_start_quat_wxyz={turn_start_quat.tolist()}",
        f"turn_end_quat_wxyz={turn_end_quat.tolist()}",
        f"yaw_rotation_deg={yaw_deg:+.6f}",
        f"tilt_deg={tilt_deg:.6f}",
        f"geodesic_rotation_deg={geodesic_deg:.6f}",
        f"thumb_sustained_last_100={'YES' if thumb_sustained else 'NO'}",
        f"final_fingers={final_fingers}",
        f"minimum_cube_center_z_m={minimum_cube_z:.9f}",
        f"maximum_palm_relative_drift_m={maximum_palm_relative_drift:.9f}",
        f"minimum_hand_cube_distance_m={minimum_hand_cube:+.9f}",
        f"minimum_finger_finger_distance_m={minimum_finger_finger:+.9f}",
        f"minimum_hand_table_distance_m={minimum_hand_table:+.9f}",
        f"drop={'YES' if drop else 'NO'}",
        f"penetration_over_0.002_m={'YES' if penetration else 'NO'}",
        f"warnings={warnings or 'none'}",
        f"abort_reason={abort_reason or 'none'}",
        f"ROTATION_GATE={'PASS' if rotation_gate else 'FAIL'}",
        f"END_HOLD_GATE={'PASS' if end_hold_gate else 'FAIL'}",
        f"SCRIPTED_TURN={'PASS' if passed else 'FAIL'}",
        "repeatability_passes=0",
        f"video={VIDEO}",
        f"video_sha256={video_hash}",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
        "",
        "phase | steps | end contact prefixes | state",
        *phase_rows,
    ]
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: lines.index("phase | steps | end contact prefixes | state")]))


if __name__ == "__main__":
    main()
