"""Batch-check LEAP poses with a 600-step cosine close and 500-step hold."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCENE = ROOT / "scenes" / "table_cube.xml"
EXPECTED_SCENE_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
MOUNT_JOINTS = ("mount_x", "mount_y", "mount_z", "mount_pitch", "mount_roll")
HAND_JOINTS = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
CLOSE_STEPS = 600
HOLD_STEPS = 500
SUSTAINED_STEPS = 100
PENETRATION_LIMIT = 0.002
CUBE_START = np.array([0.0, 0.0, 0.075])
SUMMARY_COLUMNS = (
    "pose_file", "thumb_sustained", "tip_fingers", "any_fingers",
    "final_contact", "max_disp_mm", "end_disp_mm", "ejected",
    "penetration", "GATE_A", "RELAXED",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"missing model object: {name}")
    return result


def qpos_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    return object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act")


def finger_prefix(name: str) -> str | None:
    prefix = name.split("_", 1)[0]
    return prefix if prefix in {"if", "mf", "rf", "th"} else None


def segment(name: str) -> str:
    return "tip" if name.endswith("_tip") else "non-tip"


def is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def hand_geoms(model: mujoco.MjModel) -> list[int]:
    palm = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    return [
        geom for geom in range(model.ngeom)
        if is_descendant(model, int(model.geom_bodyid[geom]), palm)
        and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    ]


def load_pose(
    path: Path, model: mujoco.MjModel
) -> tuple[dict[str, float], dict[str, float], float, list[float] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mount = payload.get("mount")
    joints = payload.get("joints")
    if not isinstance(mount, dict) or not isinstance(joints, dict):
        raise ValueError("pose must contain mount and joints objects")
    if set(mount) != set(MOUNT_JOINTS):
        raise ValueError(f"mount keys must be exactly {MOUNT_JOINTS}")
    if set(joints) != set(HAND_JOINTS):
        raise ValueError(f"joint keys must be exactly {HAND_JOINTS}")
    mount_values = {name: float(mount[name]) for name in MOUNT_JOINTS}
    joint_values = {name: float(joints[name]) for name in HAND_JOINTS}
    yaw_degrees = float(payload.get("mount_yaw_degrees", 0.0))
    if not -30.0 <= yaw_degrees <= 30.0:
        raise ValueError(f"mount_yaw_degrees={yaw_degrees} outside [-30, 30]")
    mount_quat = payload.get("mount_quat_wxyz")
    if mount_quat is not None:
        if not isinstance(mount_quat, list) or len(mount_quat) != 4:
            raise ValueError("mount_quat_wxyz must be a four-value list")
        mount_quat = [float(value) for value in mount_quat]
        if not np.isclose(np.linalg.norm(mount_quat), 1.0, atol=1e-6):
            raise ValueError("mount_quat_wxyz must be unit length")
    for name, value in mount_values.items() | joint_values.items():
        joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = model.jnt_range[joint_id]
        if model.jnt_limited[joint_id] and not lower <= value <= upper:
            raise ValueError(f"{name}={value} outside [{lower}, {upper}]")
    return mount_values, joint_values, yaw_degrees, mount_quat


def test_scene_copy(
    source: str,
    directory: Path,
    enable_yaw: bool = False,
    mount_quat: list[float] | None = None,
) -> Path:
    relative = 'file="../menagerie/leap_hand/right_hand.xml"'
    absolute = f'file="{(ROOT / "menagerie/leap_hand/right_hand.xml").resolve()}"'
    if source.count(relative) != 1:
        raise ValueError("main scene LEAP asset path is not unique")
    source = source.replace(relative, absolute)
    if mount_quat is not None:
        attach = '      <attach model="leap_hand" body="palm" prefix=""/>'
        quat = " ".join(f"{value:.12g}" for value in mount_quat)
        wrapped = (
            f'      <body name="prompt06_cache_frame" quat="{quat}">\n'
            f"{attach}\n"
            "      </body>"
        )
        if source.count(attach) != 1:
            raise ValueError("cannot add TEST-scene mount quaternion uniquely")
        source = source.replace(attach, wrapped)
    if enable_yaw:
        roll_joint = '      <joint name="mount_roll" type="hinge" axis="1 0 0" range="-0.35 0.35" damping="1"/>'
        yaw_joint = '      <joint name="mount_yaw" type="hinge" axis="0 0 1" range="-0.523599 0.523599" damping="1"/>'
        roll_actuator = '    <position name="mount_roll_act" joint="mount_roll" kp="1000" kv="60" ctrlrange="-0.35 0.35"/>'
        yaw_actuator = '    <position name="mount_yaw_act" joint="mount_yaw" kp="1000" kv="60" ctrlrange="-0.523599 0.523599"/>'
        key_values = '-0.2190 0.10 -0.10 0 0'
        if source.count(roll_joint) != 1 or source.count(roll_actuator) != 1:
            raise ValueError("cannot add TEST-scene yaw uniquely")
        if source.count(key_values) != 1:
            raise ValueError("cannot extend TEST-scene keyframe uniquely")
        source = source.replace(roll_joint, roll_joint + "\n" + yaw_joint)
        source = source.replace(roll_actuator, roll_actuator + "\n" + yaw_actuator)
        source = source.replace(key_values, '-0.2190 0.10 -0.10 0 0 0')
    path = directory / "table_cube_prompt02_test.xml"
    path.write_text(source, encoding="utf-8")
    return path


def controls(
    model: mujoco.MjModel,
    mount: dict[str, float],
    joints: dict[str, float],
    yaw_degrees: float = 0.0,
) -> np.ndarray:
    target = np.zeros(model.nu)
    for name, value in mount.items() | joints.items():
        target[actuator_id(model, name)] = value
    yaw_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "mount_yaw_act")
    if yaw_actuator >= 0:
        target[yaw_actuator] = np.deg2rad(yaw_degrees)
    return target


def geom_name(model: mujoco.MjModel, geom: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom_{geom}"


def validate(scene: Path, pose_path: Path) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    mount, target_joints, yaw_degrees, mount_quat = load_pose(pose_path, model)
    data = mujoco.MjData(model)
    key = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "start")
    mujoco.mj_resetDataKeyframe(model, data, key)
    for name, value in mount.items():
        data.qpos[qpos_address(model, name)] = value
    for name in HAND_JOINTS:
        data.qpos[qpos_address(model, name)] = 0.0
    yaw_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_yaw")
    if yaw_joint >= 0:
        data.qpos[model.jnt_qposadr[yaw_joint]] = np.deg2rad(yaw_degrees)
    open_joints = {name: 0.0 for name in HAND_JOINTS}
    open_ctrl = controls(model, mount, open_joints, yaw_degrees)
    target_ctrl = controls(model, mount, target_joints, yaw_degrees)
    data.ctrl[:] = open_ctrl
    mujoco.mj_forward(model, data)

    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    collision_hand_geoms = hand_geoms(model)
    hand_set = set(collision_hand_geoms)
    finger_pairs = [
        (a, b) for index, a in enumerate(collision_hand_geoms)
        for b in collision_hand_geoms[index + 1:]
        if finger_prefix(geom_name(model, a)) is not None
        and finger_prefix(geom_name(model, b)) is not None
        and finger_prefix(geom_name(model, a)) != finger_prefix(geom_name(model, b))
    ]
    cube_start = data.xpos[cube_body].copy()
    if not np.allclose(cube_start, CUBE_START, atol=1e-12):
        raise RuntimeError(f"cube start changed: {cube_start}")

    maximum_displacement = 0.0
    minimum_hand_cube = float("inf")
    minimum_hand_table = float("inf")
    minimum_finger_finger = float("inf")
    lowest_hand_z = float("inf")
    thumb_steps: list[bool] = []
    thumb_ever = False
    thumb_first_geom: str | None = None
    final_contacts: list[tuple[str, float]] = []
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(1, CLOSE_STEPS + HOLD_STEPS + 1):
            if step <= CLOSE_STEPS:
                alpha = 0.5 - 0.5 * np.cos(np.pi * step / CLOSE_STEPS)
                data.ctrl[:] = open_ctrl + alpha * (target_ctrl - open_ctrl)
            else:
                data.ctrl[:] = target_ctrl
            mujoco.mj_step(model, data)
            cube_position = data.xpos[cube_body]
            displacement = float(np.linalg.norm(cube_position[:2] - cube_start[:2]))
            maximum_displacement = max(maximum_displacement, displacement)

            step_contacts: list[tuple[str, float]] = []
            thumb_now = False
            for contact in data.contact[:data.ncon]:
                geoms = {int(contact.geom1), int(contact.geom2)}
                if cube_geom in geoms:
                    other = int(contact.geom2 if contact.geom1 == cube_geom else contact.geom1)
                    if other in hand_set:
                        name = geom_name(model, other)
                        distance = float(contact.dist)
                        step_contacts.append((name, distance))
                        minimum_hand_cube = min(minimum_hand_cube, distance)
                        if finger_prefix(name) == "th":
                            thumb_now = True
                            if thumb_first_geom is None:
                                thumb_first_geom = name
            thumb_steps.append(thumb_now)
            thumb_ever = thumb_ever or thumb_now
            if step == CLOSE_STEPS + HOLD_STEPS:
                final_contacts = step_contacts

            table_distances = [
                float(mujoco.mj_geomDistance(model, data, table_geom, geom, 1.0, None))
                for geom in collision_hand_geoms
            ]
            step_table = min(table_distances)
            minimum_hand_table = min(minimum_hand_table, step_table)
            lowest_hand_z = min(lowest_hand_z, 0.05 + step_table)
            if finger_pairs:
                step_fingers = min(
                    float(mujoco.mj_geomDistance(model, data, a, b, 0.01, None))
                    for a, b in finger_pairs
                )
                minimum_finger_finger = min(minimum_finger_finger, step_fingers)
    finally:
        mujoco.set_mju_user_warning(old_handler)

    solver_warnings = [
        mujoco.mjtWarning(index).name
        for index in range(mujoco.mjtWarning.mjNWARNING)
        if data.warning[index].number
    ]
    warnings.extend(solver_warnings)
    final_cube = data.xpos[cube_body].copy()
    end_displacement = float(np.linalg.norm(final_cube[:2] - cube_start[:2]))
    final_names = [name for name, _ in final_contacts]
    final_prefixes = {finger_prefix(name) for name in final_names}
    tip_fingers = sorted({
        prefix for name in final_names
        if segment(name) == "tip"
        and (prefix := finger_prefix(name)) in {"if", "mf", "rf"}
    })
    any_fingers = sorted(prefix for prefix in final_prefixes if prefix in {"if", "mf", "rf"})
    thumb_sustained = all(thumb_steps[-SUSTAINED_STEPS:])
    finger_finger_penetration = minimum_finger_finger < -PENETRATION_LIMIT
    hand_cube_penetration = minimum_hand_cube < -PENETRATION_LIMIT
    hand_table_penetration = minimum_hand_table < -PENETRATION_LIMIT
    penetration = finger_finger_penetration or hand_cube_penetration or hand_table_penetration
    safe = not penetration and not warnings
    gate_a = (
        thumb_sustained and len(tip_fingers) >= 2
        and maximum_displacement < 0.005 and end_displacement < 0.005 and safe
    )
    relaxed = (
        thumb_sustained and "th" in final_prefixes and len(any_fingers) >= 2
        and maximum_displacement < 0.015 and end_displacement < 0.015 and safe
    )
    return {
        "pose_file": pose_path.name,
        "mount": mount,
        "mount_yaw_degrees": yaw_degrees,
        "mount_quat_wxyz": mount_quat,
        "joints": target_joints,
        "thumb_ever": thumb_ever,
        "thumb_first_geom": thumb_first_geom,
        "thumb_sustained": thumb_sustained,
        "tip_fingers": tip_fingers,
        "any_fingers": any_fingers,
        "final_contacts": final_contacts,
        "final_contact": bool(final_contacts),
        "maximum_displacement": maximum_displacement,
        "end_displacement": end_displacement,
        "final_cube": final_cube,
        "ejected": not final_contacts and end_displacement > 0.015,
        "minimum_finger_finger": minimum_finger_finger,
        "minimum_hand_cube": minimum_hand_cube,
        "minimum_hand_table": minimum_hand_table,
        "lowest_hand_z": lowest_hand_z,
        "finger_finger_penetration": finger_finger_penetration,
        "hand_cube_penetration": hand_cube_penetration,
        "hand_table_penetration": hand_table_penetration,
        "penetration": penetration,
        "anti_crossing": not finger_finger_penetration,
        "warnings": warnings,
        "gate_a": gate_a,
        "relaxed": relaxed,
    }


def yes(value: object) -> str:
    return "YES" if value else "NO"


def summary_value(result: dict[str, object], column: str) -> str:
    values = {
        "pose_file": result["pose_file"],
        "thumb_sustained": yes(result["thumb_sustained"]),
        "tip_fingers": str(len(result["tip_fingers"])),
        "any_fingers": str(len(result["any_fingers"])),
        "final_contact": yes(result["final_contact"]),
        "max_disp_mm": f"{result['maximum_displacement'] * 1000:.3f}",
        "end_disp_mm": f"{result['end_displacement'] * 1000:.3f}",
        "ejected": yes(result["ejected"]),
        "penetration": yes(result["penetration"]),
        "GATE_A": "PASS" if result["gate_a"] else "FAIL",
        "RELAXED": "PASS" if result["relaxed"] else "FAIL",
    }
    return str(values[column])


def summary_table(results: list[dict[str, object]]) -> str:
    rows = [[summary_value(result, column) for column in SUMMARY_COLUMNS] for result in results]
    widths = [
        max(len(SUMMARY_COLUMNS[index]), *(len(row[index]) for row in rows))
        for index in range(len(SUMMARY_COLUMNS))
    ]
    lines = [
        " | ".join(name.ljust(widths[index]) for index, name in enumerate(SUMMARY_COLUMNS)),
        " | ".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def write_detail(result: dict[str, object], before_hash: str, after_hash: str) -> None:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(result["pose_file"])).stem)
    path = ROOT / "logs" / f"validate_poses_{safe_name}.txt"
    contacts = ";".join(
        f"{name}[dist={distance:.9f}]" for name, distance in result["final_contacts"]
    ) or "none"
    lines = [
        f"pose_file={result['pose_file']}",
        f"mount={result['mount']}",
        f"mount_yaw_degrees={result['mount_yaw_degrees']}",
        f"mount_quat_wxyz={result['mount_quat_wxyz'] or 'none'}",
        f"joints={result['joints']}",
        f"close_steps={CLOSE_STEPS}",
        "interpolation=cosine",
        f"hold_steps={HOLD_STEPS}",
        f"thumb_ever={yes(result['thumb_ever'])}",
        f"thumb_first_contact_geom={result['thumb_first_geom'] or 'none'}",
        f"thumb_sustained={yes(result['thumb_sustained'])}",
        f"tip_fingers={result['tip_fingers']}",
        f"any_fingers={result['any_fingers']}",
        f"final_contacts={contacts}",
        f"maximum_displacement_m={result['maximum_displacement']:.9f}",
        f"end_displacement_m={result['end_displacement']:.9f}",
        f"final_cube={np.array2string(result['final_cube'], precision=9, separator=', ')}",
        f"ejected={yes(result['ejected'])}",
        f"minimum_finger_finger_distance_m={result['minimum_finger_finger']:.9f}",
        f"minimum_hand_cube_distance_m={result['minimum_hand_cube']:.9f}",
        f"minimum_hand_table_distance_m={result['minimum_hand_table']:.9f}",
        f"lowest_hand_geom_z_m={result['lowest_hand_z']:.9f}",
        f"finger_finger_penetration_over_0.002_m={yes(result['finger_finger_penetration'])}",
        f"hand_cube_penetration_over_0.002_m={yes(result['hand_cube_penetration'])}",
        f"hand_table_penetration_over_0.002_m={yes(result['hand_table_penetration'])}",
        f"anti_crossing={yes(result['anti_crossing'])}",
        f"warnings={result['warnings'] or 'none'}",
        f"GATE_A={'PASS' if result['gate_a'] else 'FAIL'}",
        f"RELAXED={'PASS' if result['relaxed'] else 'FAIL'}",
        "lift_test_run=NO",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch before run: {before_hash}")
    source = MAIN_SCENE.read_text(encoding="utf-8")
    with TemporaryDirectory(prefix="prompt02_test_", dir=ROOT / "logs") as temporary:
        results = []
        for path in args.poses:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scene = test_scene_copy(
                source,
                Path(temporary),
                enable_yaw="mount_yaw_degrees" in payload,
                mount_quat=payload.get("mount_quat_wxyz"),
            )
            results.append(validate(scene, path.resolve()))
    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError(f"main scene changed during validation: {before_hash} -> {after_hash}")
    for result in results:
        write_detail(result, before_hash, after_hash)
    table = summary_table(results)
    (ROOT / "logs" / "validate_poses_summary.txt").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
