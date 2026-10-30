"""Run the first two-stage ring-finger turn from turn-ready pose T01."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import mujoco
import numpy as np

from run_scripted_turn import relative_angles_degrees
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


CANDIDATE = ROOT / "poses" / "dual_side_D01.json"
LOG = ROOT / "logs" / "prompt15_two_stage_run02_dual_side.txt"
VIDEO = ROOT / "videos" / "prompt15_two_stage_run02_dual_side.mp4"
TARGET_DEG = 90.0
TOLERANCE_DEG = 10.0
RING_SWEEP_MCP = 1.900
RING_RELEASE_ROT = -0.700


class SafetyAbort(RuntimeError):
    """Stop at the first hard contact, penetration, drift, or drop failure."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    source_pose = Path(candidate["source_held_pose"])
    source_payload = json.loads(source_pose.read_text(encoding="utf-8"))
    target_joints = {name: float(value) for name, value in candidate["target_joints"].items()}

    with TemporaryDirectory(prefix="prompt15_turn_", dir=ROOT / "logs") as temporary:
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
        data.qpos[cube_qpos : cube_qpos + 7] = (
            *source_payload["cube_start_xyz"],
            *source_payload["cube_start_quat_wxyz"],
        )
        command = mount | source_joints
        for name, value in command.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        mujoco.mj_forward(model, data)

        step_count = 0
        turn_active = False
        abort_reason: str | None = None
        minimum_hand_cube = float("inf")
        minimum_hand_table = float("inf")
        minimum_finger_finger = float("inf")
        minimum_cube_z = float("inf")
        maximum_relative_drift = 0.0
        turn_relative_start: np.ndarray | None = None
        turn_start_quat: np.ndarray | None = None
        thumb_history: list[bool] = []
        contact_history: list[set[str]] = []
        phase_rows: list[str] = []
        snapshots: list[np.ndarray] = []
        capture_every = max(1, round((1.0 / 30.0) / model.opt.timestep))
        warnings: list[str] = []

        def observe() -> set[str]:
            nonlocal minimum_hand_cube, minimum_hand_table, minimum_finger_finger
            nonlocal minimum_cube_z, maximum_relative_drift
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
            if turn_active:
                assert turn_relative_start is not None
                relative = data.xpos[cube_body] - data.xpos[palm_body]
                maximum_relative_drift = max(
                    maximum_relative_drift, float(np.linalg.norm(relative - turn_relative_start))
                )
                if "th" not in prefixes or len(prefixes & {"if", "mf", "rf"}) < 2:
                    raise SafetyAbort(
                        f"support lost at step {step_count}: prefixes={sorted(value for value in prefixes if value)}"
                    )
                deepest = min(minimum_hand_cube, minimum_hand_table, minimum_finger_finger)
                if deepest < -PENETRATION_LIMIT:
                    raise SafetyAbort(f"penetration at step {step_count}: {deepest:+.9f} m")
                if maximum_relative_drift >= 0.015:
                    raise SafetyAbort(
                        f"palm-relative drift at step {step_count}: {maximum_relative_drift:.9f} m"
                    )
                if minimum_cube_z < 0.105:
                    raise SafetyAbort(f"drop at step {step_count}: cube Z {minimum_cube_z:.9f} m")
            if step_count % capture_every == 0:
                snapshots.append(data.qpos.copy())
            return contacts

        def run_phase(name: str, target: dict[str, float], steps: int) -> None:
            nonlocal command, step_count
            start = command.copy()
            contacts: set[str] = set()
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
            prefixes = sorted(value for value in {finger_prefix(name) for name in contacts} if value)
            yaw = 0.0
            if turn_start_quat is not None:
                yaw, _, _ = relative_angles_degrees(
                    turn_start_quat, data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
                )
            phase_rows.append(
                f"{name} | {steps} | {prefixes} | yaw_deg={yaw:+.6f} | "
                f"cube_xyz={data.xpos[cube_body].tolist()}"
            )

        old_handler = mujoco.set_mju_user_warning(warnings.append)
        try:
            observe()
            run_phase("source_hold", {}, 500)
            run_phase("lift", {"mount_z": mount["mount_z"] + 0.030}, 300)
            run_phase("lift_hold", {}, 200)
            run_phase("turn_ready_transition", target_joints, int(candidate["transition_steps"]))
            run_phase("turn_ready_start_hold", {}, 500)
            turn_start_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
            turn_relative_start = data.xpos[cube_body] - data.xpos[palm_body]
            turn_active = True
            run_phase("stage1_ring_sweep", {"rf_mcp": RING_SWEEP_MCP}, 400)
            run_phase("ring_release", {"rf_rot": RING_RELEASE_ROT}, 200)
            run_phase("ring_reset", {"rf_mcp": target_joints["rf_mcp"]}, 300)
            run_phase("ring_recontact", {"rf_rot": target_joints["rf_rot"]}, 250)
            run_phase("stage2_ring_sweep", {"rf_mcp": RING_SWEEP_MCP}, 400)
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
        end_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
        yaw_deg, tilt_deg, geodesic_deg = relative_angles_degrees(turn_start_quat, end_quat)
        final_prefixes = contact_history[-1]
        final_fingers = sorted(final_prefixes & {"if", "mf", "rf"})
        thumb_sustained = all(thumb_history[-100:])
        penetration = min(minimum_hand_cube, minimum_hand_table, minimum_finger_finger) < -PENETRATION_LIMIT
        drop = minimum_cube_z < 0.105
        rotation_gate = abs(abs(yaw_deg) - TARGET_DEG) <= TOLERANCE_DEG and tilt_deg <= TOLERANCE_DEG
        end_hold_gate = abort_reason is None and thumb_sustained and len(final_fingers) >= 2
        passed = rotation_gate and end_hold_gate and not penetration and not drop and not warnings

        video_hash = "none"
        if args.video:
            VIDEO.parent.mkdir(exist_ok=True)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = source_payload["cube_start_xyz"]
            camera.lookat[2] += 0.030
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
        raise RuntimeError("main scene changed during Prompt 15")
    lines = [
        "test=Prompt 15 deterministic two-stage scripted / zero-shot turn run 02 dual-side",
        f"source_pose={source_pose}",
        f"turn_ready_candidate={CANDIDATE}",
        f"target_deg={TARGET_DEG:.1f}",
        f"tolerance_deg={TOLERANCE_DEG:.1f}",
        f"ring_sweep_mcp={RING_SWEEP_MCP:.6f}",
        f"ring_release_rot={RING_RELEASE_ROT:.6f}",
        "sampled_planner=NO",
        "learned_policy=NO",
        "MPC=NO",
        "cube_hand_attachment=NO",
        "cube_qpos_updates_during_run=0",
        f"steps={step_count}",
        f"turn_start_quat_wxyz={turn_start_quat.tolist()}",
        f"turn_end_quat_wxyz={end_quat.tolist()}",
        f"yaw_rotation_deg={yaw_deg:+.6f}",
        f"tilt_deg={tilt_deg:.6f}",
        f"geodesic_rotation_deg={geodesic_deg:.6f}",
        f"maximum_palm_relative_drift_m={maximum_relative_drift:.9f}",
        f"thumb_sustained_last_100={'YES' if thumb_sustained else 'NO'}",
        f"final_fingers={final_fingers}",
        f"minimum_cube_center_z_m={minimum_cube_z:.9f}",
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
        f"video={VIDEO if args.video else 'none'}",
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
