"""Bounded amplitude scan of the measured O01 ring-rotation torque direction."""

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


POSE = ROOT / "poses" / "oriented_grasp_O01.json"
OUTPUT_LOG = ROOT / "logs" / "prompt17_rf_rot_amplitude_scan.txt"
REQUESTED_AMPLITUDES = (0.10, 0.11, 0.12, 0.13, 0.14)
RAMP_STEPS = 180
HOLD_STEPS = 150
SUPPORT_GRACE = 12


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
    payload = json.loads(POSE.read_text())
    rows: list[dict[str, object]] = []

    with TemporaryDirectory(prefix="prompt17_amplitude_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(MAIN_SCENE.read_text(), Path(temporary), enable_yaw=True)
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, joints, yaw_degrees, _ = load_pose(POSE, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        data.qpos[cube_qpos : cube_qpos + 7] = (*payload["cube_start_xyz"], *payload["cube_start_quat_wxyz"])
        for name, value in mount.items() | joints.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        yaw = math.radians(yaw_degrees)
        data.qpos[qpos_address(model, "mount_yaw")] = yaw
        data.ctrl[actuator_id(model, "mount_yaw")] = yaw
        mujoco.mj_forward(model, data)
        for step in range(800):
            phase = min((step + 1) / 300, 1.0)
            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
            data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
            mujoco.mj_step(model, data)

        base_qpos, base_qvel = data.qpos.copy(), data.qvel.copy()
        base_act, base_ctrl, base_time = data.act.copy(), data.ctrl.copy(), data.time
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        hand = hand_geoms(model)
        hand_set = set(hand)
        finger_pairs = [
            (a, b) for index, a in enumerate(hand) for b in hand[index + 1 :]
            if finger_prefix(geom_name(model, a)) in {"if", "mf", "rf", "th"}
            and finger_prefix(geom_name(model, b)) in {"if", "mf", "rf", "th"}
            and finger_prefix(geom_name(model, a)) != finger_prefix(geom_name(model, b))
        ]
        rf_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "rf_rot")
        rf_lower = float(model.jnt_range[rf_joint, 0])
        available = joints["rf_rot"] - rf_lower
        amplitudes = sorted(set((*REQUESTED_AMPLITUDES, available)))

        for requested in amplitudes:
            amplitude = min(requested, available)
            target = joints["rf_rot"] - amplitude
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.act[:] = base_act
            data.ctrl[:] = base_ctrl
            data.time = base_time
            mujoco.mj_forward(model, data)
            start_position = data.xpos[cube_body].copy()
            start_quaternion = data.xquat[cube_body].copy()
            maximum_drift = 0.0
            deepest_cube = deepest_table = deepest_self = float("inf")
            unsupported = 0
            stop_reason = "completed"
            final_contacts: set[str] = set()
            warnings: list[str] = []
            old_handler = mujoco.set_mju_user_warning(warnings.append)
            try:
                for step in range(RAMP_STEPS + HOLD_STEPS):
                    if step < RAMP_STEPS:
                        phase = (step + 1) / RAMP_STEPS
                        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                        data.ctrl[actuator_id(model, "rf_rot")] = joints["rf_rot"] - amplitude * blend
                    mujoco.mj_step(model, data)
                    maximum_drift = max(maximum_drift, float(np.linalg.norm(data.xpos[cube_body] - start_position)))
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
                    prefixes = {finger_prefix(name) for name in contacts}
                    supported = "th" in prefixes and len(prefixes & {"if", "mf", "rf"}) >= 2
                    unsupported = 0 if supported else unsupported + 1
                    final_contacts = contacts
                    if unsupported >= SUPPORT_GRACE:
                        stop_reason = "support_loss"
                        break
                    if maximum_drift >= 0.015:
                        stop_reason = "drift_limit"
                        break
                    if min(deepest_cube, deepest_table, deepest_self) < -PENETRATION_LIMIT:
                        stop_reason = "penetration"
                        break
            finally:
                mujoco.set_mju_user_warning(old_handler)
            if warnings and stop_reason == "completed":
                stop_reason = "warning"
            yaw_rotation, tilt, geodesic = relative_angles_degrees(start_quaternion, data.xquat[cube_body])
            rows.append({
                "requested": requested, "amplitude": amplitude, "target": target,
                "yaw": yaw_rotation, "tilt": tilt, "geodesic": geodesic,
                "drift": maximum_drift, "stop": stop_reason, "contacts": sorted(final_contacts),
                "deepest_cube": deepest_cube, "deepest_table": deepest_table,
                "deepest_self": deepest_self, "warnings": warnings,
            })

    safe_rows = [row for row in rows if row["stop"] == "completed"]
    best = max(safe_rows, key=lambda row: abs(float(row["yaw"])), default=None)
    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during amplitude scan")
    lines = [
        "test=Prompt 17 TEST-only bounded rf_rot amplitude scan",
        f"source_pose={POSE}", f"rf_rot_start_rad={joints['rf_rot']:.9f}",
        f"rf_rot_lower_limit_rad={rf_lower:.9f}", f"available_negative_amplitude_rad={available:.9f}",
        f"ramp_steps={RAMP_STEPS}", f"hold_steps={HOLD_STEPS}",
        "cube_hand_attachment=NO", "cube_qpos_updates_during_run=0", "full_90_degree_turn=NO",
        f"best_safe_amplitude_rad={best['amplitude'] if best else 'none'}",
        f"best_safe_yaw_deg={best['yaw'] if best else 'none'}",
        f"best_safe_tilt_deg={best['tilt'] if best else 'none'}",
        f"best_safe_geodesic_deg={best['geodesic'] if best else 'none'}",
        f"best_safe_drift_mm={best['drift'] * 1000 if best else 'none'}",
        f"main_scene_sha256_before={before_hash}", f"main_scene_sha256_after={after_hash}", "",
        "amplitude_rad | target_rad | yaw_deg | tilt_deg | geodesic_deg | drift_mm | stop | final_contacts",
    ]
    lines.extend(
        f"{row['amplitude']:.9f} | {row['target']:.9f} | {row['yaw']:+.6f} | "
        f"{row['tilt']:.6f} | {row['geodesic']:.6f} | "
        f"{row['drift'] * 1000:.3f} | {row['stop']} | {row['contacts']}"
        for row in rows
    )
    OUTPUT_LOG.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("")]))
    if best:
        print(f"best_contacts={best['contacts']}")


if __name__ == "__main__":
    main()
