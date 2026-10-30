"""Plan ring-fingertip core support from the validated lifted grasp."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import MAIN_SCENE, RING_JOINTS, SCENE, SOURCE_POSE
from validate_poses import (
    ROOT,
    actuator_id,
    geom_name,
    load_pose,
    object_id,
    qpos_address,
    sha256,
)


OUTPUT_LOG = ROOT / "logs" / "prompt38_initial_ring_core_planner.txt"
SAMPLES = 30_000
SEED = 38001


def replay_lifted_grasp(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    payload = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    mount, joints, _, _ = load_pose(SOURCE_POSE, model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    data.qpos[cube_qpos : cube_qpos + 7] = (
        *payload["cube_start_xyz"],
        *payload["cube_start_quat_wxyz"],
    )
    data.qpos[int(model.jnt_qposadr[face_joint])] = float(payload["face_turn_rad"])
    for name, value in mount.items() | joints.items():
        data.qpos[qpos_address(model, name)] = value
        data.ctrl[actuator_id(model, name)] = value
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    for step in range(800):
        phase = min((step + 1) / 300, 1.0)
        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
        data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
        mujoco.mj_step(model, data)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_lifted_grasp(model, data)
    base_qpos = data.qpos.copy()
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    tip_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    ring_geoms = {
        geom
        for geom in range(model.ngeom)
        if (geom_name(model, geom) or "").startswith("rf_")
        and "visual" not in (geom_name(model, geom) or "")
    }
    ranges = {
        name: tuple(
            map(
                float,
                model.jnt_range[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)],
            )
        )
        for name in RING_JOINTS
    }
    starts = {name: float(base_qpos[qpos_address(model, name)]) for name in RING_JOINTS}
    rng = np.random.default_rng(SEED)
    candidates: list[dict[str, object]] = []
    near_core = 0
    safe = 0
    core_fromto = np.empty(6)
    face_fromto = np.empty(6)
    for _ in range(SAMPLES):
        targets = {name: float(rng.uniform(*ranges[name])) for name in RING_JOINTS}
        data.qpos[:] = base_qpos
        for name, value in targets.items():
            data.qpos[qpos_address(model, name)] = value
        mujoco.mj_forward(model, data)
        core_distance = float(
            mujoco.mj_geomDistance(model, data, core_geom, tip_geom, 0.20, core_fromto)
        )
        if not -0.001 <= core_distance <= 0.004:
            continue
        near_core += 1
        face_distance = float(
            mujoco.mj_geomDistance(model, data, face_geom, tip_geom, 0.20, face_fromto)
        )
        crossed = False
        for contact in data.contact[: data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            if first not in ring_geoms and second not in ring_geoms:
                continue
            other = second if first in ring_geoms else first
            if other not in {core_geom, face_geom} and float(contact.dist) < -0.002:
                crossed = True
                break
        if crossed:
            continue
        safe += 1
        motion = sum((targets[name] - starts[name]) ** 2 for name in RING_JOINTS) ** 0.5
        score = abs(core_distance + 0.0002) + 0.0001 * motion
        candidates.append(
            {
                "score": score,
                "core_distance_m": core_distance,
                "face_distance_m": face_distance,
                "motion_norm_rad": motion,
                "targets": targets,
            }
        )
    candidates.sort(key=lambda candidate: float(candidate["score"]))
    lines = [
        "test=initial lifted-grasp ring-core planner",
        f"samples={SAMPLES}",
        f"seed={SEED}",
        f"ring_start={json.dumps(starts, sort_keys=True)}",
        f"near_core_count={near_core}",
        f"safe_count={safe}",
        *(f"candidate={json.dumps(candidate, sort_keys=True)}" for candidate in candidates[:30]),
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
