"""Scan TEST-only fixed base quaternions while preserving the O01 local grasp."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from run_scripted_turn import quat_multiply, relative_angles_degrees
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


SOURCE_POSE = ROOT / "poses" / "oriented_grasp_O01.json"
OUTPUT_LOG = ROOT / "logs" / "prompt19_fixed_base_quaternion_scan.txt"
GRID_DEGREES = (-30, -20, -10, 0, 10, 20, 30)
LOCAL_PITCH_DEGREES = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
LOCAL_ROLL_DEGREES = (5.0, 7.5, 10.0, 12.5, 15.0)
PROBE_DELTAS = (-0.10, -0.12, -0.14270996867816654)
MOUNT_PITCH = -0.23333333333333334
MOUNT_ROLL = -0.11666666666666667
SETTLE_STEPS = 200
LIFT_STEPS = 300
RAMP_STEPS = 160
HOLD_STEPS = 80
SUPPORT_GRACE = 12


def axis_quaternion(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    return np.array([math.cos(half), *(math.sin(half) * np.asarray(axis))])


def main() -> None:
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(SOURCE_POSE.read_text())
    source = MAIN_SCENE.read_text()
    rows: list[dict[str, object]] = []

    with TemporaryDirectory(prefix="prompt19_basequat_", dir=ROOT / "logs") as temporary:
        temporary_path = Path(temporary)
        reference_directory = temporary_path / "reference"
        reference_directory.mkdir()
        reference_scene = test_scene_copy(source, reference_directory, enable_yaw=True)
        reference_model = mujoco.MjModel.from_xml_path(str(reference_scene))
        reference_data = mujoco.MjData(reference_model)
        reference_mount, joints, yaw_degrees, _ = load_pose(SOURCE_POSE, reference_model)
        reference_cube_joint = object_id(reference_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        reference_cube_qpos = int(reference_model.jnt_qposadr[reference_cube_joint])
        reference_palm = object_id(reference_model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        reference_data.qpos[reference_cube_qpos : reference_cube_qpos + 7] = (
            *payload["cube_start_xyz"], *payload["cube_start_quat_wxyz"],
        )
        for name, value in reference_mount.items() | joints.items():
            reference_data.qpos[qpos_address(reference_model, name)] = value
        reference_data.qpos[qpos_address(reference_model, "mount_yaw")] = math.radians(yaw_degrees)
        mujoco.mj_forward(reference_model, reference_data)
        palm_position = reference_data.xpos[reference_palm].copy()
        palm_rotation = reference_data.xmat[reference_palm].reshape(3, 3).copy()
        cube_rotation = np.empty(9)
        mujoco.mju_quat2Mat(cube_rotation, np.asarray(payload["cube_start_quat_wxyz"]))
        relative_position = palm_rotation.T @ (np.asarray(payload["cube_start_xyz"]) - palm_position)
        relative_rotation = palm_rotation.T @ cube_rotation.reshape(3, 3)

        cases = [(float(pitch), float(roll), -0.10) for pitch in GRID_DEGREES for roll in GRID_DEGREES]
        cases.extend(
            (pitch, roll, delta)
            for pitch in LOCAL_PITCH_DEGREES
            for roll in LOCAL_ROLL_DEGREES
            for delta in PROBE_DELTAS
        )
        for case_index, (base_pitch_degrees, base_roll_degrees, probe_delta) in enumerate(cases):
                base_quaternion = quat_multiply(
                    axis_quaternion((0.0, 1.0, 0.0), base_pitch_degrees),
                    axis_quaternion((1.0, 0.0, 0.0), base_roll_degrees),
                )
                case_directory = temporary_path / f"case_{case_index:03d}"
                case_directory.mkdir()
                scene = test_scene_copy(
                    source, case_directory, enable_yaw=True, mount_quat=base_quaternion.tolist()
                )
                model = mujoco.MjModel.from_xml_path(str(scene))
                data = mujoco.MjData(model)
                cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
                cube_qpos = int(model.jnt_qposadr[cube_joint])
                cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
                cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
                table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
                palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
                hand = hand_geoms(model)
                hand_set = set(hand)
                mount = reference_mount | {"mount_pitch": MOUNT_PITCH, "mount_roll": MOUNT_ROLL}
                for name, value in mount.items() | joints.items():
                    data.qpos[qpos_address(model, name)] = value
                    data.ctrl[actuator_id(model, name)] = value
                yaw = math.radians(yaw_degrees)
                data.qpos[qpos_address(model, "mount_yaw")] = yaw
                data.ctrl[actuator_id(model, "mount_yaw")] = yaw
                mujoco.mj_forward(model, data)
                candidate_palm_rotation = data.xmat[palm_body].reshape(3, 3)
                candidate_cube_position = data.xpos[palm_body] + candidate_palm_rotation @ relative_position
                candidate_cube_rotation = candidate_palm_rotation @ relative_rotation
                candidate_cube_quaternion = np.empty(4)
                mujoco.mju_mat2Quat(candidate_cube_quaternion, candidate_cube_rotation.ravel())
                data.qpos[cube_qpos : cube_qpos + 7] = (
                    *candidate_cube_position, *candidate_cube_quaternion,
                )
                mujoco.mj_forward(model, data)

                initial_relative = data.xpos[cube_body] - data.xpos[palm_body]
                maximum_drift = 0.0
                deepest_cube = deepest_table = deepest_self = float("inf")
                unsupported = 0
                failure = "none"
                turn_start_quaternion: np.ndarray | None = None
                final_contacts: set[str] = set()
                finger_pairs = [
                    (a, b) for index, a in enumerate(hand) for b in hand[index + 1 :]
                    if finger_prefix(geom_name(model, a)) in {"if", "mf", "rf", "th"}
                    and finger_prefix(geom_name(model, b)) in {"if", "mf", "rf", "th"}
                    and finger_prefix(geom_name(model, a)) != finger_prefix(geom_name(model, b))
                ]

                for step in range(SETTLE_STEPS + LIFT_STEPS + RAMP_STEPS + HOLD_STEPS):
                    if SETTLE_STEPS <= step < SETTLE_STEPS + LIFT_STEPS:
                        phase = (step - SETTLE_STEPS + 1) / LIFT_STEPS
                        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                        data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
                    if step == SETTLE_STEPS + LIFT_STEPS:
                        turn_start_quaternion = data.xquat[cube_body].copy()
                    if SETTLE_STEPS + LIFT_STEPS <= step < SETTLE_STEPS + LIFT_STEPS + RAMP_STEPS:
                        phase = (step - SETTLE_STEPS - LIFT_STEPS + 1) / RAMP_STEPS
                        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                        data.ctrl[actuator_id(model, "rf_rot")] = joints["rf_rot"] + probe_delta * blend
                    mujoco.mj_step(model, data)
                    relative = data.xpos[cube_body] - data.xpos[palm_body]
                    maximum_drift = max(maximum_drift, float(np.linalg.norm(relative - initial_relative)))
                    contacts: set[str] = set()
                    for contact in data.contact[: data.ncon]:
                        first, second = int(contact.geom1), int(contact.geom2)
                        pair = {first, second}
                        if cube_geom in pair:
                            other = second if first == cube_geom else first
                            if other in hand_set:
                                contacts.add(geom_name(model, other))
                                deepest_cube = min(deepest_cube, float(contact.dist))
                        elif table_geom in pair:
                            other = second if first == table_geom else first
                            if other in hand_set:
                                deepest_table = min(deepest_table, float(contact.dist))
                        elif first in hand_set and second in hand_set:
                            a, b = finger_prefix(geom_name(model, first)), finger_prefix(geom_name(model, second))
                            if a in {"if", "mf", "rf", "th"} and b in {"if", "mf", "rf", "th"} and a != b:
                                deepest_self = min(deepest_self, float(contact.dist))
                    turn_active = step >= SETTLE_STEPS + LIFT_STEPS
                    prefixes = {finger_prefix(name) for name in contacts}
                    supported = "th" in prefixes and len(prefixes & {"if", "mf", "rf"}) >= 2
                    unsupported = (0 if supported else unsupported + 1) if turn_active else 0
                    final_contacts = contacts
                    if turn_active and unsupported >= SUPPORT_GRACE:
                        failure = "support_loss"
                        break
                    if maximum_drift >= 0.015:
                        failure = "drift_limit"
                        break
                    if min(deepest_cube, deepest_table, deepest_self) < -PENETRATION_LIMIT:
                        failure = "penetration"
                        break

                if turn_start_quaternion is None:
                    yaw_rotation, tilt, geodesic = 0.0, 180.0, 180.0
                else:
                    yaw_rotation, tilt, geodesic = relative_angles_degrees(
                        turn_start_quaternion, data.xquat[cube_body]
                    )
                warnings = sum(int(data.warning[index].number) for index in range(mujoco.mjtWarning.mjNWARNING))
                if warnings and failure == "none":
                    failure = "warning"
                gate = failure == "none" and abs(yaw_rotation) >= 1.0 and abs(yaw_rotation) > tilt
                rows.append({
                    "base_pitch": base_pitch_degrees, "base_roll": base_roll_degrees,
                    "probe_delta": probe_delta,
                    "base_quaternion": base_quaternion, "cube_position": candidate_cube_position,
                    "cube_quaternion": candidate_cube_quaternion, "yaw": yaw_rotation,
                    "tilt": tilt, "geodesic": geodesic, "drift": maximum_drift,
                    "failure": failure, "gate": gate, "contacts": sorted(final_contacts),
                    "deepest_cube": deepest_cube, "deepest_table": deepest_table,
                    "deepest_self": deepest_self,
                })

    rows.sort(
        key=lambda row: (
            bool(row["gate"]), row["failure"] == "none",
            abs(float(row["yaw"])) - float(row["tilt"]), -float(row["drift"]),
        ),
        reverse=True,
    )
    passing = [row for row in rows if row["gate"]]
    unique_passing: list[dict[str, object]] = []
    seen_orientations: set[tuple[float, float]] = set()
    for row in passing:
        orientation = (float(row["base_pitch"]), float(row["base_roll"]))
        if orientation in seen_orientations:
            continue
        seen_orientations.add(orientation)
        unique_passing.append(row)
    pose_paths: list[Path] = []
    for index, row in enumerate(unique_passing[:3], start=1):
        path = ROOT / "poses" / f"fixed_base_axis_Q{index:02d}.json"
        path.write_text(json.dumps({
            "test_only": True, "label": f"fixed_base_axis_Q{index:02d}",
            "cube_start_xyz": row["cube_position"].tolist(),
            "cube_start_quat_wxyz": row["cube_quaternion"].tolist(),
            "mount": reference_mount | {"mount_pitch": MOUNT_PITCH, "mount_roll": MOUNT_ROLL},
            "mount_yaw_degrees": yaw_degrees,
            "mount_quat_wxyz": row["base_quaternion"].tolist(), "joints": joints,
            "target_axis_probe": {"joint": "rf_rot", "delta_rad": row["probe_delta"],
                                  "screen_yaw_deg": row["yaw"], "screen_tilt_deg": row["tilt"],
                                  "screen_geodesic_deg": row["geodesic"],
                                  "screen_drift_m": row["drift"], "pass": True},
            "planner": {"kind": "fixed base-quaternion grid (not MPC, not learned)",
                        "coarse_grid_degrees": list(GRID_DEGREES),
                        "local_pitch_degrees": list(LOCAL_PITCH_DEGREES),
                        "local_roll_degrees": list(LOCAL_ROLL_DEGREES),
                        "probe_deltas_rad": list(PROBE_DELTAS)},
        }, indent=2) + "\n")
        pose_paths.append(path)

    best = rows[0]
    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during fixed-base scan")
    lines = [
        "test=Prompt 19 TEST-only fixed base-quaternion scan",
        f"grid_degrees={GRID_DEGREES}", f"samples={len(rows)}", "full_90_degree_turn=NO",
        f"local_pitch_degrees={LOCAL_PITCH_DEGREES}",
        f"local_roll_degrees={LOCAL_ROLL_DEGREES}", f"probe_deltas_rad={PROBE_DELTAS}",
        "local_grasp_geometry=O01 palm-relative cube pose preserved",
        f"passing_candidates={len(passing)}", f"unique_passing_orientations={len(unique_passing)}",
        f"pose_files={[str(path) for path in pose_paths] or 'none'}",
        f"best_base_pitch_deg={best['base_pitch']}", f"best_base_roll_deg={best['base_roll']}",
        f"best_probe_delta_rad={best['probe_delta']}",
        f"best_yaw_deg={best['yaw']:+.9f}", f"best_tilt_deg={best['tilt']:.9f}",
        f"best_geodesic_deg={best['geodesic']:.9f}", f"best_drift_mm={best['drift'] * 1000:.6f}",
        f"best_failure={best['failure']}", f"best_gate={'PASS' if best['gate'] else 'FAIL'}",
        f"main_scene_sha256_before={before_hash}", f"main_scene_sha256_after={after_hash}", "",
        "base_pitch_deg | base_roll_deg | probe_delta_rad | yaw_deg | tilt_deg | geodesic_deg | drift_mm | failure | gate",
    ]
    lines.extend(
        f"{row['base_pitch']:+g} | {row['base_roll']:+g} | {row['probe_delta']:+.9f} | {row['yaw']:+.6f} | "
        f"{row['tilt']:.6f} | {row['geodesic']:.6f} | {row['drift'] * 1000:.3f} | "
        f"{row['failure']} | {'PASS' if row['gate'] else 'FAIL'}"
        for row in rows
    )
    OUTPUT_LOG.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("")]))


if __name__ == "__main__":
    main()
