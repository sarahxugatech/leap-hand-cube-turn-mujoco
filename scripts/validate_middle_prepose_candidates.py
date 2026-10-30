"""Dynamically screen planned middle-finger pre-contact poses over 50 steps."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from plan_middle_handoff import MIDDLE_JOINTS, OUTPUT_LOG as PLANNER_LOG
from search_articulated_face_followthrough import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    SCENE,
)
from search_articulated_face_kf03 import replay_to_kf02
from validate_poses import (
    PENETRATION_LIMIT,
    ROOT,
    actuator_id,
    finger_prefix,
    geom_name,
    hand_geoms,
    object_id,
    qpos_address,
    sha256,
)


OUTPUT_LOG = ROOT / "logs" / "prompt34_middle_prepose_dynamic_screen.txt"
TRANSITION_STEPS = 50
HOLD_STEPS = 50


def load_candidates() -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("candidate="))
        for line in PLANNER_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("candidate=")
    ]


def evaluate(model: mujoco.MjModel, candidate: dict[str, object]) -> dict[str, object]:
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    starts = {name: float(data.qpos[qpos_address(model, name)]) for name in MIDDLE_JOINTS}
    targets = {name: float(value) for name, value in candidate["targets"].items()}
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    mf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "mf_ds_collision_1")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    hand_set = set(hand_geoms(model))
    baseline_cube = data.xpos[cube_body].copy()
    baseline_angle = float(data.qpos[face_qpos])
    maximum_drift = np.zeros(3)
    minimum_distance = float("inf")
    thumb_index_support_steps = 0
    ring_face_steps = 0
    middle_face_steps = 0
    maximum_middle_force = 0.0
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(TRANSITION_STEPS + HOLD_STEPS):
            blend = 1.0
            if step < TRANSITION_STEPS:
                blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / TRANSITION_STEPS)
            for name in MIDDLE_JOINTS:
                data.ctrl[actuator_id(model, name)] = starts[name] + blend * (
                    targets[name] - starts[name]
                )
            mujoco.mj_step(model, data)
            prefixes: set[str] = set()
            ring_face = False
            middle_face = False
            middle_force = 0.0
            for contact_index, contact in enumerate(data.contact[: data.ncon]):
                first, second = int(contact.geom1), int(contact.geom2)
                pair = {first, second}
                if first in hand_set or second in hand_set:
                    minimum_distance = min(minimum_distance, float(contact.dist))
                if core_geom in pair:
                    other = second if first == core_geom else first
                    if other in hand_set:
                        prefix = finger_prefix(geom_name(model, other))
                        if prefix:
                            prefixes.add(prefix)
                if face_geom in pair:
                    other = second if first == face_geom else first
                    ring_face |= other == rf_geom
                    if other == mf_geom:
                        middle_face = True
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, contact_index, force)
                        middle_force += max(0.0, float(force[0]))
            thumb_index_support_steps += {"th", "if"}.issubset(prefixes)
            ring_face_steps += ring_face
            middle_face_steps += middle_face
            maximum_middle_force = max(maximum_middle_force, middle_force)
            maximum_drift = np.maximum(
                maximum_drift, np.abs(data.xpos[cube_body] - baseline_cube)
            )
    finally:
        mujoco.set_mju_user_warning(old_handler)
    fromto = np.empty(6)
    final_gap = float(
        mujoco.mj_geomDistance(model, data, face_geom, mf_geom, 0.20, fromto)
    )
    final_targets_error = {
        name: targets[name] - float(data.qpos[qpos_address(model, name)])
        for name in MIDDLE_JOINTS
    }
    total = TRANSITION_STEPS + HOLD_STEPS
    result = {
        "planner_score": candidate["score"],
        "targets": targets,
        "final_face_gap_m": final_gap,
        "final_face_angle_deg": math.degrees(float(data.qpos[face_qpos])),
        "face_angle_change_deg": math.degrees(float(data.qpos[face_qpos]) - baseline_angle),
        "maximum_xyz_drift_mm": [1000 * float(value) for value in maximum_drift],
        "maximum_axis_drift_mm": 1000 * float(np.max(maximum_drift)),
        "thumb_index_support_rate": thumb_index_support_steps / total,
        "ring_face_contact_rate": ring_face_steps / total,
        "middle_face_contact_rate": middle_face_steps / total,
        "maximum_middle_force_n": maximum_middle_force,
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": minimum_distance < -PENETRATION_LIMIT,
        "final_tracking_errors_rad": final_targets_error,
        "warnings": warnings,
    }
    result["prepose_pass"] = (
        0.0005 <= final_gap <= 0.003
        and abs(float(result["face_angle_change_deg"])) <= 0.2
        and float(result["maximum_axis_drift_mm"]) < 0.1
        and float(result["thumb_index_support_rate"]) == 1.0
        and float(result["ring_face_contact_rate"]) >= 0.95
        and not bool(result["penetration_over_0.002_m"])
        and not warnings
    )
    return result


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before dynamic prepose screen")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    candidates = load_candidates()
    results = [evaluate(model, candidate) for candidate in candidates]
    passing = sorted(
        (result for result in results if result["prepose_pass"]),
        key=lambda result: abs(float(result["final_face_gap_m"]) - 0.0015),
    )
    ranked = sorted(
        results,
        key=lambda result: (
            abs(float(result["final_face_gap_m"]) - 0.0015),
            float(result["maximum_axis_drift_mm"]),
        ),
    )
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during dynamic prepose screen")
    lines = [
        "test=Prompt 34 middle-finger 50-step dynamic prepose screen",
        f"candidate_count={len(candidates)}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_passing={json.dumps(passing[0], sort_keys=True) if passing else 'NONE'}",
        f"closest_dynamic={json.dumps(ranked[0], sort_keys=True)}",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
