"""Find kinematically reachable middle-distal pre-contact poses at KF02."""

from __future__ import annotations

import json

import mujoco
import numpy as np

from search_articulated_face_followthrough import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    SCENE,
)
from search_articulated_face_kf03 import replay_to_kf02
from validate_poses import ROOT, geom_name, object_id, qpos_address, sha256


OUTPUT_LOG = ROOT / "logs" / "prompt34_middle_handoff_planner.txt"
MIDDLE_JOINTS = ("mf_mcp", "mf_rot", "mf_pip", "mf_dip")
SAMPLES = 20_000
SEED = 34001
TARGET_GAP_M = 0.0015


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before middle-handoff planning")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    base_qpos = data.qpos.copy()
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    mf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "mf_ds_collision_1")
    middle_geoms = {
        geom
        for geom in range(model.ngeom)
        if (geom_name(model, geom) or "").startswith("mf_")
        and "visual" not in (geom_name(model, geom) or "")
    }
    joint_ranges = {}
    for name in MIDDLE_JOINTS:
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ranges[name] = tuple(map(float, model.jnt_range[joint]))
    starts = {name: float(base_qpos[qpos_address(model, name)]) for name in MIDDLE_JOINTS}
    rng = np.random.default_rng(SEED)
    candidates: list[dict[str, object]] = []
    reachable_side_count = 0
    safe_side_count = 0
    fromto = np.empty(6)
    core_fromto = np.empty(6)
    for _ in range(SAMPLES):
        targets = {
            name: float(rng.uniform(*joint_ranges[name])) for name in MIDDLE_JOINTS
        }
        data.qpos[:] = base_qpos
        for name, value in targets.items():
            data.qpos[qpos_address(model, name)] = value
        mujoco.mj_forward(model, data)
        distance = float(mujoco.mj_geomDistance(model, data, face_geom, mf_geom, 0.20, fromto))
        if not -0.001 <= distance <= 0.012:
            continue
        local_face_point = data.xmat[face_body].reshape(3, 3).T @ (
            fromto[:3] - data.xpos[face_body]
        )
        on_side = (
            max(abs(local_face_point[0]), abs(local_face_point[1])) >= 0.022
            and abs(local_face_point[2]) <= 0.009
        )
        if not on_side:
            continue
        reachable_side_count += 1
        excessive_crossing = False
        for contact in data.contact[: data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            if first not in middle_geoms and second not in middle_geoms:
                continue
            if {first, second} == {face_geom, mf_geom}:
                continue
            other = second if first in middle_geoms else first
            if other not in {face_geom, core_geom} and float(contact.dist) < -0.002:
                excessive_crossing = True
                break
        if excessive_crossing:
            continue
        core_distance = float(
            mujoco.mj_geomDistance(model, data, core_geom, mf_geom, 0.20, core_fromto)
        )
        if core_distance < -0.001:
            continue
        safe_side_count += 1
        motion = sum((targets[name] - starts[name]) ** 2 for name in MIDDLE_JOINTS)
        score = abs(distance - TARGET_GAP_M) + 0.0001 * motion
        candidates.append(
            {
                "score": score,
                "face_distance_m": distance,
                "core_distance_m": core_distance,
                "face_point_local": [float(value) for value in local_face_point],
                "targets": targets,
                "motion_norm_rad": motion**0.5,
            }
        )
    candidates.sort(key=lambda candidate: float(candidate["score"]))
    top = candidates[:20]
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during middle-handoff planning")
    lines = [
        "test=Prompt 34 middle-finger kinematic handoff planner",
        f"samples={SAMPLES}",
        f"seed={SEED}",
        f"middle_start={json.dumps(starts, sort_keys=True)}",
        f"reachable_side_count={reachable_side_count}",
        f"safe_side_count={safe_side_count}",
        *(f"candidate={json.dumps(candidate, sort_keys=True)}" for candidate in top),
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
