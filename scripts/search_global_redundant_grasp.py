"""Search test-only initial topologies with ring-tip contact across the face seam."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import MAIN_SCENE, RING_JOINTS, SCENE, SOURCE_POSE
from validate_poses import (
    PENETRATION_LIMIT,
    ROOT,
    actuator_id,
    geom_name,
    hand_geoms,
    load_pose,
    object_id,
    qpos_address,
    sha256,
)


OUTPUT_LOG = ROOT / "logs" / "prompt39_global_redundant_grasp_planner.txt"
EXPECTED_MAIN_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
EXPECTED_SCENE_SHA256 = "baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb"
SEED = 39_001
UNIFORM_SAMPLES = 20_000
GUIDED_SAMPLES = 40_000
KEEP = 40
DUAL_MIN_M = -0.002
DUAL_MAX_M = 0.0015
SUPPORT_MAX_M = 0.002


def quat_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def collision_geoms(model: mujoco.MjModel, prefix: str) -> list[int]:
    return [
        geom
        for geom in hand_geoms(model)
        if (geom_name(model, geom) or "").startswith(prefix)
        and "visual" not in (geom_name(model, geom) or "")
    ]


def geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first: int,
    second: int,
) -> float:
    fromto = np.empty(6)
    return float(mujoco.mj_geomDistance(model, data, first, second, 0.25, fromto))


def minimum_distance_to_group(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: int,
    geoms: list[int],
) -> float:
    return min(geom_distance(model, data, target, geom) for geom in geoms)


def minimum_relevant_contact(data: mujoco.MjData, relevant: set[int]) -> float:
    distances = [
        float(contact.dist)
        for contact in data.contact[: data.ncon]
        if int(contact.geom1) in relevant or int(contact.geom2) in relevant
    ]
    return min(distances, default=float("inf"))


def main() -> None:
    if sha256(MAIN_SCENE) != EXPECTED_MAIN_SHA256:
        raise RuntimeError("frozen main scene hash mismatch")
    if sha256(SCENE) != EXPECTED_SCENE_SHA256:
        raise RuntimeError("articulated test scene hash mismatch")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    source = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    mount, joints, _, _ = load_pose(SOURCE_POSE, model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    ring_tip = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    support_geoms = {
        prefix: collision_geoms(model, prefix) for prefix in ("th_", "if_", "mf_")
    }
    relevant = set(hand_geoms(model)) | {core_geom, face_geom}
    ranges = {
        name: tuple(
            map(
                float,
                model.jnt_range[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)],
            )
        )
        for name in RING_JOINTS
    }
    base_cube = np.asarray(source["cube_start_xyz"], dtype=float)
    rng = np.random.default_rng(SEED)
    ranked: list[dict[str, object]] = []
    dual_count = 0
    support_count = 0
    safe_count = 0

    def evaluate(sample_index: int, guided: bool) -> None:
        nonlocal dual_count, support_count, safe_count
        pitch = float(rng.uniform(-0.2, 0.2))
        roll = float(rng.uniform(-0.2, 0.2))
        yaw = float(rng.uniform(-0.2, 0.2))
        ring = {name: float(rng.uniform(*ranges[name])) for name in RING_JOINTS}

        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[cube_qpos : cube_qpos + 7] = (*base_cube, *quat_yaw(yaw))
        data.qpos[face_qpos] = float(source["face_turn_rad"])
        candidate_mount = mount | {"mount_pitch": pitch, "mount_roll": roll}
        candidate_joints = joints | ring
        for name, value in candidate_mount.items() | candidate_joints.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        mujoco.mj_forward(model, data)

        if guided:
            tip_xy = np.asarray(data.geom_xpos[ring_tip, :2])
            axis = int(rng.integers(0, 2))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            offset = np.zeros(2)
            offset[axis] = sign * float(rng.uniform(0.019, 0.031))
            offset[1 - axis] = float(rng.uniform(-0.020, 0.020))
            cube_xy = tip_xy + offset + rng.uniform(-0.002, 0.002, size=2)
            delta_xy = cube_xy - base_cube[:2]
            if np.any(np.abs(delta_xy) > 0.045):
                return
        else:
            delta_xy = rng.uniform(-0.045, 0.045, size=2)
            cube_xy = base_cube[:2] + delta_xy
        data.qpos[cube_qpos : cube_qpos + 2] = cube_xy
        mujoco.mj_forward(model, data)

        core_distance = geom_distance(model, data, core_geom, ring_tip)
        face_distance = geom_distance(model, data, face_geom, ring_tip)
        dual_error = max(
            max(DUAL_MIN_M - core_distance, 0.0, core_distance - DUAL_MAX_M),
            max(DUAL_MIN_M - face_distance, 0.0, face_distance - DUAL_MAX_M),
        )
        dual = dual_error == 0.0
        dual_count += dual
        support_distances = {
            prefix.removesuffix("_"): minimum_distance_to_group(
                model, data, core_geom, geoms
            )
            for prefix, geoms in support_geoms.items()
        }
        support_error = sum(max(distance - SUPPORT_MAX_M, 0.0) for distance in support_distances.values())
        support = support_error == 0.0
        support_count += dual and support
        minimum_contact = minimum_relevant_contact(data, relevant)
        safe = minimum_contact >= -PENETRATION_LIMIT
        safe_count += dual and support and safe
        motion = math.sqrt(sum((ring[name] - joints[name]) ** 2 for name in RING_JOINTS))
        penetration_error = max(-PENETRATION_LIMIT - minimum_contact, 0.0)
        score = (
            1000.0 * dual_error
            + 100.0 * support_error
            + 1000.0 * penetration_error
            + 0.02 * motion
        )
        ranked.append(
            {
                "sample": sample_index,
                "guided": guided,
                "score": score,
                "static_gate": dual and support and safe,
                "dual_contact": dual,
                "support_proximity": support,
                "penetration_safe": safe,
                "cube_xyz": [float(cube_xy[0]), float(cube_xy[1]), float(base_cube[2])],
                "cube_quat_wxyz": list(quat_yaw(yaw)),
                "cube_delta_xy_m": [float(value) for value in delta_xy],
                "relative_yaw_rad": yaw,
                "mount": {name: float(value) for name, value in candidate_mount.items()},
                "joints": {name: float(value) for name, value in candidate_joints.items()},
                "ring_core_distance_m": core_distance,
                "ring_face_distance_m": face_distance,
                "support_core_distances_m": support_distances,
                "minimum_relevant_contact_m": minimum_contact,
                "ring_motion_norm_rad": motion,
            }
        )

    for index in range(UNIFORM_SAMPLES):
        evaluate(index, False)
    for index in range(UNIFORM_SAMPLES, UNIFORM_SAMPLES + GUIDED_SAMPLES):
        evaluate(index, True)

    ranked.sort(
        key=lambda item: (
            not bool(item["static_gate"]),
            not (bool(item["dual_contact"]) and bool(item["penetration_safe"])),
            not bool(item["penetration_safe"]),
            float(item["score"]),
        )
    )
    selected = ranked[:KEEP]
    lines = [
        "test=global redundant-grasp static topology search",
        f"seed={SEED}",
        f"uniform_samples={UNIFORM_SAMPLES}",
        f"guided_samples={GUIDED_SAMPLES}",
        "cube_xy_delta_range_m=[-0.045,+0.045]",
        "mount_pitch_roll_absolute_range_rad=[-0.2,+0.2]",
        "relative_yaw_range_rad=[-0.2,+0.2] (cube yaw; scene has no mount_yaw joint)",
        f"ring_joint_ranges={json.dumps(ranges, sort_keys=True)}",
        f"dual_contact_window_m=[{DUAL_MIN_M},{DUAL_MAX_M}]",
        f"support_distance_max_m={SUPPORT_MAX_M}",
        f"dual_count={dual_count}",
        f"dual_plus_support_count={support_count}",
        f"static_gate_count={safe_count}",
        *(f"candidate={json.dumps(item, sort_keys=True)}" for item in selected),
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    for item in selected[:5]:
        print(f"candidate={json.dumps(item, sort_keys=True)}")


if __name__ == "__main__":
    main()
