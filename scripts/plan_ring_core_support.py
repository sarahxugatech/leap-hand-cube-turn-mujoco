"""Plan ring-fingertip poses near the cube core from the KF03 endpoint."""

from __future__ import annotations

import json

import mujoco
import numpy as np

from search_articulated_face_followthrough import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    RING_JOINTS,
    SCENE,
)
from validate_four_finger_reverse import replay_to_kf03
from validate_poses import ROOT, geom_name, object_id, qpos_address, sha256


OUTPUT_LOG = ROOT / "logs" / "prompt37_ring_core_support_planner.txt"
SAMPLES = 50_000
SEED = 37001
TARGET_GAP_M = 0.0005


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before ring-core planning")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_to_kf03(model, data)
    base_qpos = data.qpos.copy()
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
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
            mujoco.mj_geomDistance(model, data, core_geom, rf_geom, 0.20, core_fromto)
        )
        if not -0.001 <= core_distance <= 0.006:
            continue
        near_core += 1
        face_distance = float(
            mujoco.mj_geomDistance(model, data, face_geom, rf_geom, 0.20, face_fromto)
        )
        if face_distance < 0.001:
            continue
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
        score = abs(core_distance - TARGET_GAP_M) + 0.00015 * motion
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
        "test=ring-fingertip core-support kinematic planner from KF03",
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
