"""Search a natural ring posture plus thumb/index/middle redundant core support."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import MAIN_SCENE, RING_JOINTS, SCENE, SOURCE_POSE
from search_global_redundant_grasp import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    collision_geoms,
    geom_distance,
    minimum_relevant_contact,
    quat_yaw,
)
from validate_poses import (
    PENETRATION_LIMIT,
    ROOT,
    actuator_id,
    hand_geoms,
    load_pose,
    object_id,
    qpos_address,
    sha256,
)


OUTPUT_LOG = ROOT / "logs" / "prompt40_natural_redundant_grasp_planner.txt"
SEED = 40_001
RING_SAMPLES = 80_000
SUPPORT_SAMPLES = 800
RING_KEEP = 24
OUTPUT_KEEP = 20
CONTACT_MIN_M = -0.002
CONTACT_MAX_M = 0.0015
SUPPORT_FINGERS = {
    "th": ("th_cmc", "th_axl", "th_mcp", "th_ipl"),
    "if": ("if_mcp", "if_rot", "if_pip", "if_dip"),
    "mf": ("mf_mcp", "mf_rot", "mf_pip", "mf_dip"),
}


def natural_ring_sample(rng: np.random.Generator) -> dict[str, float]:
    mcp = float(rng.uniform(0.0, 1.8))
    pip = float(rng.uniform(-0.1, 1.5))
    dip_low = max(-0.1, pip - 0.7)
    dip_high = min(1.6, pip + 0.7)
    return {
        "rf_mcp": mcp,
        "rf_rot": float(rng.uniform(-0.5, 0.5)),
        "rf_pip": pip,
        "rf_dip": float(rng.uniform(dip_low, dip_high)),
    }


def sample_support(
    model: mujoco.MjModel,
    rng: np.random.Generator,
    prefix: str,
    names: tuple[str, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in names:
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = map(float, model.jnt_range[joint])
        if prefix in {"if", "mf"} and name.endswith("_rot"):
            lower, upper = max(lower, -0.5), min(upper, 0.5)
        result[name] = float(rng.uniform(lower, upper))
    return result


def set_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_qpos: int,
    face_qpos: int,
    cube_xyz: list[float],
    cube_quat: list[float],
    mount: dict[str, float],
    joints: dict[str, float],
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[cube_qpos : cube_qpos + 7] = (*cube_xyz, *cube_quat)
    data.qpos[face_qpos] = 0.0
    for name, value in mount.items() | joints.items():
        data.qpos[qpos_address(model, name)] = value
        data.ctrl[actuator_id(model, name)] = value
    mujoco.mj_forward(model, data)


def window_error(distance: float) -> float:
    return max(CONTACT_MIN_M - distance, 0.0, distance - CONTACT_MAX_M)


def main() -> None:
    if sha256(MAIN_SCENE) != EXPECTED_MAIN_SHA256 or sha256(SCENE) != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    source = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    base_mount, base_joints, _, _ = load_pose(SOURCE_POSE, model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    ring_tip = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    ring_geoms = collision_geoms(model, "rf_")
    relevant = set(hand_geoms(model)) | {core_geom, face_geom}
    base_cube = np.asarray(source["cube_start_xyz"], dtype=float)
    rng = np.random.default_rng(SEED)
    ring_candidates: list[dict[str, object]] = []
    ring_near_raw = 0
    ring_penetration_rejects = 0

    for sample in range(RING_SAMPLES):
        pitch = float(rng.uniform(-0.2, 0.2))
        roll = float(rng.uniform(-0.2, 0.2))
        yaw = float(rng.uniform(-0.2, 0.2))
        ring = natural_ring_sample(rng)
        mount = base_mount | {"mount_pitch": pitch, "mount_roll": roll}
        set_candidate(
            model,
            data,
            cube_qpos,
            face_qpos,
            [*base_cube],
            list(quat_yaw(yaw)),
            mount,
            base_joints | ring,
        )
        tip_xy = np.asarray(data.geom_xpos[ring_tip, :2])
        axis = int(rng.integers(0, 2))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        offset = np.zeros(2)
        offset[axis] = sign * float(rng.uniform(0.019, 0.031))
        offset[1 - axis] = float(rng.uniform(-0.020, 0.020))
        cube_xy = tip_xy + offset + rng.uniform(-0.002, 0.002, size=2)
        delta_xy = cube_xy - base_cube[:2]
        if np.any(np.abs(delta_xy) > 0.045):
            continue
        cube_z = float(data.geom_xpos[ring_tip, 2] - 0.008333333333 + rng.uniform(-0.010, 0.010))
        delta_z = cube_z - float(base_cube[2])
        if abs(delta_z) > 0.030:
            continue
        cube_xyz = [float(cube_xy[0]), float(cube_xy[1]), cube_z]
        cube_quat = list(quat_yaw(yaw))
        set_candidate(
            model,
            data,
            cube_qpos,
            face_qpos,
            cube_xyz,
            cube_quat,
            mount,
            base_joints | ring,
        )
        core_distances = {
            geom: geom_distance(model, data, core_geom, geom) for geom in ring_geoms
        }
        face_distances = {
            geom: geom_distance(model, data, face_geom, geom) for geom in ring_geoms
        }
        core_contact_geom = min(core_distances, key=core_distances.get)
        face_contact_geom = min(face_distances, key=face_distances.get)
        core_distance = core_distances[core_contact_geom]
        face_distance = face_distances[face_contact_geom]
        ring_object_minimum = min(
            geom_distance(model, data, target, geom)
            for target in (core_geom, face_geom)
            for geom in ring_geoms
        )
        error = window_error(core_distance) + window_error(face_distance)
        if error > 0.003:
            continue
        ring_near_raw += 1
        if ring_object_minimum < -PENETRATION_LIMIT:
            ring_penetration_rejects += 1
            continue
        ring_candidates.append(
            {
                "sample": sample,
                "ring_error_m": error,
                "cube_xyz": cube_xyz,
                "cube_quat_wxyz": cube_quat,
                "cube_delta_xy_m": [float(value) for value in delta_xy],
                "cube_delta_z_m": delta_z,
                "relative_yaw_rad": yaw,
                "mount": mount,
                "joints": base_joints | ring,
                "ring_core_distance_m": core_distance,
                "ring_face_distance_m": face_distance,
                "ring_core_geom": model.geom(core_contact_geom).name,
                "ring_face_geom": model.geom(face_contact_geom).name,
                "ring_object_minimum_m": ring_object_minimum,
            }
        )

    ring_candidates.sort(key=lambda item: float(item["ring_error_m"]))
    solved: list[dict[str, object]] = []
    for candidate in ring_candidates[:RING_KEEP]:
        joints = dict(candidate["joints"])
        support_distances: dict[str, float] = {}
        for prefix, names in SUPPORT_FINGERS.items():
            geoms = collision_geoms(model, f"{prefix}_")
            best: tuple[float, float, dict[str, float]] | None = None
            for _ in range(SUPPORT_SAMPLES):
                targets = sample_support(model, rng, prefix, names)
                trial_joints = joints | targets
                set_candidate(
                    model,
                    data,
                    cube_qpos,
                    face_qpos,
                    candidate["cube_xyz"],
                    candidate["cube_quat_wxyz"],
                    candidate["mount"],
                    trial_joints,
                )
                distance = min(geom_distance(model, data, core_geom, geom) for geom in geoms)
                object_minimum = min(
                    geom_distance(model, data, target, geom)
                    for target in (core_geom, face_geom)
                    for geom in geoms
                )
                error = window_error(distance)
                penetration_error = max(-PENETRATION_LIMIT - object_minimum, 0.0)
                motion = math.sqrt(sum((targets[name] - base_joints[name]) ** 2 for name in names))
                rank = (error + 10.0 * penetration_error, motion, targets)
                if best is None or rank[:2] < best[:2]:
                    best = rank
            assert best is not None
            joints.update(best[2])
            support_distances[prefix] = min(
                geom_distance(model, data, core_geom, geom) for geom in geoms
            )

        set_candidate(
            model,
            data,
            cube_qpos,
            face_qpos,
            candidate["cube_xyz"],
            candidate["cube_quat_wxyz"],
            candidate["mount"],
            joints,
        )
        ring_core = min(geom_distance(model, data, core_geom, geom) for geom in ring_geoms)
        ring_face = min(geom_distance(model, data, face_geom, geom) for geom in ring_geoms)
        support_distances = {
            prefix: min(
                geom_distance(model, data, core_geom, geom)
                for geom in collision_geoms(model, f"{prefix}_")
            )
            for prefix in SUPPORT_FINGERS
        }
        minimum_contact = minimum_relevant_contact(data, relevant)
        static_gate = (
            window_error(ring_core) == 0.0
            and window_error(ring_face) == 0.0
            and all(window_error(value) == 0.0 for value in support_distances.values())
            and minimum_contact >= -PENETRATION_LIMIT
        )
        score = (
            window_error(ring_core)
            + window_error(ring_face)
            + sum(window_error(value) for value in support_distances.values())
            + max(-PENETRATION_LIMIT - minimum_contact, 0.0)
        )
        solved.append(
            candidate
            | {
                "score": score,
                "static_gate": static_gate,
                "joints": joints,
                "ring_core_distance_m": ring_core,
                "ring_face_distance_m": ring_face,
                "support_core_distances_m": support_distances,
                "minimum_relevant_contact_m": minimum_contact,
            }
        )

    solved.sort(key=lambda item: (not bool(item["static_gate"]), float(item["score"])))
    lines = [
        "test=natural four-finger redundant-grasp static search",
        f"seed={SEED}",
        f"ring_samples={RING_SAMPLES}",
        f"support_samples_per_finger={SUPPORT_SAMPLES}",
        "ring_rot_range_rad=[-0.5,+0.5]",
        "ring_mcp_range_rad=[0.0,1.8]",
        "ring_pip_range_rad=[-0.1,1.5]",
        "ring_dip_within_0.7_rad_of_pip=true",
        "ring_dual_contact=separate ring collision segments allowed",
        "cube_relative_z_range_m=[-0.03,+0.03]",
        f"ring_near_raw_count={ring_near_raw}",
        f"ring_penetration_reject_count={ring_penetration_rejects}",
        f"ring_near_candidate_count={len(ring_candidates)}",
        f"support_solved_count={len(solved)}",
        f"static_gate_count={sum(bool(item['static_gate']) for item in solved)}",
        *(f"candidate={json.dumps(item, sort_keys=True)}" for item in solved[:OUTPUT_KEEP]),
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:11]))
    for item in solved[:5]:
        print(f"candidate={json.dumps(item, sort_keys=True)}")


if __name__ == "__main__":
    main()
