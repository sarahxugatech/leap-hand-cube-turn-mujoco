"""Search a new held topology with the true ring tip on the movable face."""

from __future__ import annotations

import json

import mujoco
import numpy as np

from search_articulated_face_followthrough import MAIN_SCENE, SCENE, SOURCE_POSE
from search_global_redundant_grasp import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    collision_geoms,
    geom_distance,
    minimum_relevant_contact,
    quat_yaw,
)
from search_natural_redundant_grasp import (
    CONTACT_MAX_M,
    CONTACT_MIN_M,
    SUPPORT_FINGERS,
    natural_ring_sample,
    sample_support,
    set_candidate,
    window_error,
)
from validate_poses import PENETRATION_LIMIT, ROOT, hand_geoms, load_pose, object_id, sha256


OUTPUT_LOG = ROOT / "logs" / "prompt41_global_natural_ring_tip_grasp.txt"
SEED = 41_101
TOPOLOGY_SAMPLES = 100_000
SUPPORT_SAMPLES = 1_000
TOPOLOGY_KEEP = 30
OUTPUT_KEEP = 20


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
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    tip_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_tip")
    ring_geoms = collision_geoms(model, "rf_")
    relevant = set(hand_geoms(model)) | {core_geom, face_geom}
    base_cube = np.asarray(source["cube_start_xyz"], dtype=float)
    rng = np.random.default_rng(SEED)
    topologies: list[dict[str, object]] = []
    raw_tip_near = 0
    ring_penetration_rejects = 0

    for sample in range(TOPOLOGY_SAMPLES):
        mount = base_mount | {
            "mount_pitch": float(rng.uniform(-0.2, 0.2)),
            "mount_roll": float(rng.uniform(-0.2, 0.2)),
        }
        yaw = float(rng.uniform(-0.2, 0.2))
        ring = natural_ring_sample(rng)
        cube_quat = list(quat_yaw(yaw))
        set_candidate(
            model, data, cube_qpos, face_qpos, [*base_cube], cube_quat, mount, base_joints | ring
        )
        tip = np.asarray(data.geom_xpos[tip_geom])
        axis = int(rng.integers(0, 2))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        offset = np.zeros(2)
        offset[axis] = sign * float(rng.uniform(0.032, 0.042))
        offset[1 - axis] = float(rng.uniform(-0.018, 0.018))
        cube_xy = tip[:2] + offset + rng.uniform(-0.002, 0.002, size=2)
        cube_z = float(tip[2] - 0.016666666667 + rng.uniform(-0.008, 0.008))
        delta = np.array([cube_xy[0], cube_xy[1], cube_z]) - base_cube
        if np.any(np.abs(delta[:2]) > 0.045) or abs(delta[2]) > 0.030:
            continue
        cube_xyz = [float(cube_xy[0]), float(cube_xy[1]), cube_z]
        set_candidate(
            model, data, cube_qpos, face_qpos, cube_xyz, cube_quat, mount, base_joints | ring
        )
        fromto = np.empty(6)
        tip_distance = float(
            mujoco.mj_geomDistance(model, data, face_geom, tip_geom, 0.25, fromto)
        )
        face_local = data.xmat[face_body].reshape(3, 3).T @ (fromto[:3] - data.xpos[face_body])
        on_side = max(abs(face_local[0]), abs(face_local[1])) >= 0.023 and abs(face_local[2]) <= 0.009
        ring_minimum = min(
            geom_distance(model, data, target, geom)
            for target in (core_geom, face_geom)
            for geom in ring_geoms
        )
        if not on_side or window_error(tip_distance) > 0.0:
            continue
        raw_tip_near += 1
        if ring_minimum < -PENETRATION_LIMIT:
            ring_penetration_rejects += 1
            continue
        topologies.append(
            {
                "sample": sample,
                "cube_xyz": cube_xyz,
                "cube_quat_wxyz": cube_quat,
                "cube_delta_xyz_m": [float(value) for value in delta],
                "mount": mount,
                "joints": base_joints | ring,
                "tip_face_distance_m": tip_distance,
                "ring_object_minimum_m": ring_minimum,
            }
        )

    topologies.sort(key=lambda item: abs(float(item["tip_face_distance_m"]) + 0.0002))
    solved: list[dict[str, object]] = []
    for candidate in topologies[:TOPOLOGY_KEEP]:
        joints = dict(candidate["joints"])
        for prefix, names in SUPPORT_FINGERS.items():
            geoms = collision_geoms(model, f"{prefix}_")
            best: tuple[float, float, dict[str, float]] | None = None
            for _ in range(SUPPORT_SAMPLES):
                targets = sample_support(model, rng, prefix, names)
                set_candidate(
                    model,
                    data,
                    cube_qpos,
                    face_qpos,
                    candidate["cube_xyz"],
                    candidate["cube_quat_wxyz"],
                    candidate["mount"],
                    joints | targets,
                )
                core_distance = min(
                    geom_distance(model, data, core_geom, geom) for geom in geoms
                )
                object_minimum = min(
                    geom_distance(model, data, target, geom)
                    for target in (core_geom, face_geom)
                    for geom in geoms
                )
                rank = (
                    window_error(core_distance)
                    + 10.0 * max(-PENETRATION_LIMIT - object_minimum, 0.0),
                    sum((targets[name] - base_joints[name]) ** 2 for name in names),
                    targets,
                )
                if best is None or rank[:2] < best[:2]:
                    best = rank
            assert best is not None
            joints.update(best[2])

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
        support_distances = {
            prefix: min(
                geom_distance(model, data, core_geom, geom)
                for geom in collision_geoms(model, f"{prefix}_")
            )
            for prefix in SUPPORT_FINGERS
        }
        tip_distance = geom_distance(model, data, face_geom, tip_geom)
        minimum_contact = minimum_relevant_contact(data, relevant)
        static_gate = (
            CONTACT_MIN_M <= tip_distance <= CONTACT_MAX_M
            and all(CONTACT_MIN_M <= value <= CONTACT_MAX_M for value in support_distances.values())
            and minimum_contact >= -PENETRATION_LIMIT
        )
        score = (
            window_error(tip_distance)
            + sum(window_error(value) for value in support_distances.values())
            + max(-PENETRATION_LIMIT - minimum_contact, 0.0)
        )
        solved.append(
            candidate
            | {
                "score": score,
                "static_gate": static_gate,
                "joints": joints,
                "tip_face_distance_m": tip_distance,
                "support_core_distances_m": support_distances,
                "minimum_relevant_contact_m": minimum_contact,
            }
        )

    solved.sort(key=lambda item: (not bool(item["static_gate"]), float(item["score"])))
    lines = [
        "test=global natural true-ring-tip held-grasp search",
        f"seed={SEED}",
        f"topology_samples={TOPOLOGY_SAMPLES}",
        f"support_samples_per_finger={SUPPORT_SAMPLES}",
        "target_geom=rf_tip",
        "rf_rot_range_rad=[-0.5,+0.5]",
        f"raw_tip_near_count={raw_tip_near}",
        f"ring_penetration_reject_count={ring_penetration_rejects}",
        f"raw_tip_topology_count={len(topologies)}",
        f"support_solved_count={len(solved)}",
        f"static_gate_count={sum(bool(item['static_gate']) for item in solved)}",
        *(f"candidate={json.dumps(item, sort_keys=True)}" for item in solved[:OUTPUT_KEEP]),
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:11]))
    for item in solved[:3]:
        print(f"candidate={json.dumps(item, sort_keys=True)}")


if __name__ == "__main__":
    main()
