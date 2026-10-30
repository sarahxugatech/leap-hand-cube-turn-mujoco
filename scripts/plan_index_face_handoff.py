"""Plan index-fingertip pre-contact poses on the face layer from KF03."""

from __future__ import annotations

import json

import mujoco
import numpy as np

from search_articulated_face_followthrough import MAIN_SCENE, SCENE
from validate_four_finger_reverse import replay_to_kf03
from validate_poses import ROOT, geom_name, object_id, qpos_address, sha256


OUTPUT_LOG = ROOT / "logs" / "prompt37_index_face_handoff_planner.txt"
INDEX_JOINTS = ("if_mcp", "if_rot", "if_pip", "if_dip")
SAMPLES = 20_000
SEED = 37002


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_to_kf03(model, data)
    base_qpos = data.qpos.copy()
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    tip_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "if_ds_collision_1")
    index_geoms = {
        geom
        for geom in range(model.ngeom)
        if (geom_name(model, geom) or "").startswith("if_")
        and "visual" not in (geom_name(model, geom) or "")
    }
    ranges = {
        name: tuple(
            map(
                float,
                model.jnt_range[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)],
            )
        )
        for name in INDEX_JOINTS
    }
    starts = {name: float(base_qpos[qpos_address(model, name)]) for name in INDEX_JOINTS}
    rng = np.random.default_rng(SEED)
    candidates: list[dict[str, object]] = []
    near_side = 0
    safe = 0
    fromto = np.empty(6)
    core_fromto = np.empty(6)
    for _ in range(SAMPLES):
        targets = {name: float(rng.uniform(*ranges[name])) for name in INDEX_JOINTS}
        data.qpos[:] = base_qpos
        for name, value in targets.items():
            data.qpos[qpos_address(model, name)] = value
        mujoco.mj_forward(model, data)
        distance = float(
            mujoco.mj_geomDistance(model, data, face_geom, tip_geom, 0.20, fromto)
        )
        if not -0.001 <= distance <= 0.008:
            continue
        local_point = data.xmat[face_body].reshape(3, 3).T @ (
            fromto[:3] - data.xpos[face_body]
        )
        if not (
            max(abs(local_point[0]), abs(local_point[1])) >= 0.022
            and abs(local_point[2]) <= 0.009
        ):
            continue
        near_side += 1
        core_distance = float(
            mujoco.mj_geomDistance(model, data, core_geom, tip_geom, 0.20, core_fromto)
        )
        if core_distance < -0.001:
            continue
        crossed = False
        for contact in data.contact[: data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            if first not in index_geoms and second not in index_geoms:
                continue
            other = second if first in index_geoms else first
            if other not in {face_geom, core_geom} and float(contact.dist) < -0.002:
                crossed = True
                break
        if crossed:
            continue
        safe += 1
        motion = sum((targets[name] - starts[name]) ** 2 for name in INDEX_JOINTS) ** 0.5
        score = abs(distance - 0.0015) + 0.0001 * motion
        candidates.append(
            {
                "score": score,
                "face_distance_m": distance,
                "core_distance_m": core_distance,
                "face_point_local": [float(value) for value in local_point],
                "motion_norm_rad": motion,
                "targets": targets,
            }
        )
    candidates.sort(key=lambda candidate: float(candidate["score"]))
    lines = [
        "test=index-fingertip face pre-contact planner from KF03",
        f"samples={SAMPLES}",
        f"seed={SEED}",
        f"index_start={json.dumps(starts, sort_keys=True)}",
        f"near_side_count={near_side}",
        f"safe_count={safe}",
        *(f"candidate={json.dumps(candidate, sort_keys=True)}" for candidate in candidates[:30]),
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
