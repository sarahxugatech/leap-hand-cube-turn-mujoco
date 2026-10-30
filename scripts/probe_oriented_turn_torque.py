"""Probe small non-thumb joint motions for cube torque from an oriented lifted grasp."""

from __future__ import annotations

import argparse
import json
import math
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
    sha256,
    test_scene_copy,
)


POSE = ROOT / "poses" / "oriented_grasp_O01.json"
OUTPUT_LOG = ROOT / "logs" / "prompt16_oriented_torque_probe.txt"
LIFT_M = 0.030
PROBE_DELTA = 0.10
RAMP_STEPS = 150
HOLD_STEPS = 100
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=Path, default=POSE)
    parser.add_argument("--log", type=Path, default=OUTPUT_LOG)
    parser.add_argument("--label", default="Prompt 16")
    parser.add_argument("--delta", type=float, default=PROBE_DELTA)
    parser.add_argument("--joint", choices=HAND_JOINTS)
    parser.add_argument("--transition", type=Path)
    args = parser.parse_args()
    pose = args.pose.resolve()
    output_log = args.log.resolve()
    transition = args.transition.resolve() if args.transition else None
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(pose.read_text())
    source = MAIN_SCENE.read_text()
    rows: list[dict[str, object]] = []

    with TemporaryDirectory(prefix="joint_probe_", dir=ROOT / "logs") as temporary:
        yaw_enabled = "mount_yaw_degrees" in payload
        scene = test_scene_copy(
            source,
            Path(temporary),
            enable_yaw=yaw_enabled,
            mount_quat=payload.get("mount_quat_wxyz"),
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, joints, yaw_degrees, _ = load_pose(pose, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        data.qpos[cube_qpos : cube_qpos + 7] = (*payload["cube_start_xyz"], *payload["cube_start_quat_wxyz"])
        for name, value in mount.items() | joints.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        if yaw_enabled:
            yaw = math.radians(yaw_degrees)
            data.qpos[qpos_address(model, "mount_yaw")] = yaw
            data.ctrl[actuator_id(model, "mount_yaw")] = yaw
        mujoco.mj_forward(model, data)
        for step in range(800):
            phase = min((step + 1) / 300, 1.0)
            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
            data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + LIFT_M * blend
            mujoco.mj_step(model, data)
        if transition:
            transition_payload = json.loads(transition.read_text())
            if Path(transition_payload["source_held_pose"]).resolve() != pose:
                raise ValueError("transition source_held_pose does not match --pose")
            target_joints = transition_payload["target_joints"]
            for step in range(int(transition_payload["transition_steps"])):
                phase = (step + 1) / int(transition_payload["transition_steps"])
                blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                for name in HAND_JOINTS:
                    data.ctrl[actuator_id(model, name)] = joints[name] + (
                        target_joints[name] - joints[name]
                    ) * blend
                mujoco.mj_step(model, data)
            for _ in range(500):
                mujoco.mj_step(model, data)
            joints = target_joints

        base_qpos, base_qvel = data.qpos.copy(), data.qvel.copy()
        base_act, base_ctrl, base_time = data.act.copy(), data.ctrl.copy(), data.time
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

        probe_joints = [args.joint] if args.joint else [
            name for name in HAND_JOINTS if not name.startswith("th_")
        ]
        if any(name.startswith("th_") for name in probe_joints):
            raise ValueError("thumb joints are not valid support-preserving probes")
        for joint_name in probe_joints:
            joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            lower, upper = model.jnt_range[joint_id]
            for direction in (-1.0, 1.0):
                target = float(np.clip(joints[joint_name] + direction * args.delta, lower, upper))
                actual_delta = target - joints[joint_name]
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

                for step in range(RAMP_STEPS + HOLD_STEPS):
                    if step < RAMP_STEPS:
                        phase = (step + 1) / RAMP_STEPS
                        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                        data.ctrl[actuator_id(model, joint_name)] = joints[joint_name] + actual_delta * blend
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

                yaw_rotation, tilt, geodesic = relative_angles_degrees(start_quaternion, data.xquat[cube_body])
                rows.append({
                    "joint": joint_name, "delta": actual_delta, "yaw": yaw_rotation,
                    "tilt": tilt, "geodesic": geodesic,
                    "drift": maximum_drift, "stop": stop_reason,
                    "contacts": sorted(final_contacts), "deepest_cube": deepest_cube,
                    "deepest_table": deepest_table, "deepest_self": deepest_self,
                    "safe": stop_reason == "completed",
                })

    safe_rows = [row for row in rows if row["safe"]]
    safe_rows.sort(key=lambda row: abs(float(row["yaw"])), reverse=True)
    best = safe_rows[0] if safe_rows else None
    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during torque probe")
    lines = [
        f"test={args.label} TEST-only small-joint torque probe",
        f"source_pose={pose}", f"probe_delta_rad={args.delta}",
        f"transition={transition or 'none'}",
        f"ramp_steps={RAMP_STEPS}", f"hold_steps={HOLD_STEPS}",
        "cube_hand_attachment=NO", "cube_qpos_updates_during_run=0", "full_90_degree_turn=NO",
        f"safe_completed={len(safe_rows)}/{len(rows)}",
        f"best_safe={best['joint'] if best else 'none'} delta={best['delta'] if best else 'none'} "
        f"yaw_deg={best['yaw'] if best else 'none'} tilt_deg={best['tilt'] if best else 'none'} "
        f"geodesic_deg={best['geodesic'] if best else 'none'} drift_mm={best['drift'] * 1000 if best else 'none'}",
        f"main_scene_sha256_before={before_hash}", f"main_scene_sha256_after={after_hash}", "",
        "joint | delta_rad | yaw_deg | tilt_deg | geodesic_deg | drift_mm | stop | final_contacts",
    ]
    lines.extend(
        f"{row['joint']} | {row['delta']:+.6f} | {row['yaw']:+.6f} | {row['tilt']:.6f} | "
        f"{row['geodesic']:.6f} | "
        f"{row['drift'] * 1000:.3f} | {row['stop']} | {row['contacts']}"
        for row in sorted(rows, key=lambda item: (str(item["joint"]), float(item["delta"])))
    )
    output_log.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("")]))
    if best:
        print(f"best_contacts={best['contacts']}")


if __name__ == "__main__":
    main()
