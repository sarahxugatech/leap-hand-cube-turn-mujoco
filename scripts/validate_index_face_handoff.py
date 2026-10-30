"""Dynamically screen index pre-contact transitions while middle stays on core."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from plan_index_face_handoff import INDEX_JOINTS, OUTPUT_LOG as PLANNER_LOG
from search_articulated_face_followthrough import SCENE
from validate_four_finger_reverse import replay_to_kf03
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


OUTPUT_LOG = ROOT / "logs" / "prompt37_index_face_handoff_dynamic_screen.txt"
OUTPUT_POSE = ROOT / "poses" / "articulated_index_face_precontact_R01.json"
TRANSITION_STEPS = 100
HOLD_STEPS = 100


def load_candidates() -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("candidate="))
        for line in PLANNER_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("candidate=")
    ]


def evaluate(model: mujoco.MjModel, candidate: dict[str, object]) -> dict[str, object]:
    data = mujoco.MjData(model)
    replay_to_kf03(model, data)
    starts = {name: float(data.qpos[qpos_address(model, name)]) for name in INDEX_JOINTS}
    targets = {name: float(value) for name, value in candidate["targets"].items()}
    cube = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    tip_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "if_ds_collision_1")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    hand_set = set(hand_geoms(model))
    baseline_relative = data.xpos[cube] - data.xpos[palm]
    baseline_angle = float(data.qpos[face_qpos])
    maximum_drift = np.zeros(3)
    minimum_distance = float("inf")
    thumb_middle_support = 0
    ring_face = 0
    index_face_history: list[bool] = []
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(TRANSITION_STEPS + HOLD_STEPS):
            blend = 1.0
            if step < TRANSITION_STEPS:
                blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / TRANSITION_STEPS)
            for name in INDEX_JOINTS:
                data.ctrl[actuator_id(model, name)] = starts[name] + blend * (
                    targets[name] - starts[name]
                )
            mujoco.mj_step(model, data)
            prefixes: set[str] = set()
            ring_on_face = False
            index_on_face = False
            for contact in data.contact[: data.ncon]:
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
                    ring_on_face |= other == rf_geom
                    index_on_face |= other == tip_geom
            thumb_middle_support += {"th", "mf"}.issubset(prefixes)
            ring_face += ring_on_face
            index_face_history.append(index_on_face)
            relative = data.xpos[cube] - data.xpos[palm]
            maximum_drift = np.maximum(maximum_drift, np.abs(relative - baseline_relative))
    finally:
        mujoco.set_mju_user_warning(old_handler)
    fromto = np.empty(6)
    final_gap = float(
        mujoco.mj_geomDistance(model, data, face_geom, tip_geom, 0.20, fromto)
    )
    total = TRANSITION_STEPS + HOLD_STEPS
    result = {
        "targets": targets,
        "final_face_gap_m": final_gap,
        "final_face_angle_deg": math.degrees(float(data.qpos[face_qpos])),
        "face_angle_change_deg": math.degrees(float(data.qpos[face_qpos]) - baseline_angle),
        "maximum_xyz_drift_mm": [1000 * float(value) for value in maximum_drift],
        "maximum_axis_drift_mm": 1000 * float(np.max(maximum_drift)),
        "thumb_middle_support_rate": thumb_middle_support / total,
        "ring_face_contact_rate": ring_face / total,
        "index_face_contact_rate": sum(index_face_history) / total,
        "index_face_last50_rate": sum(index_face_history[-50:]) / 50,
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": minimum_distance < -PENETRATION_LIMIT,
        "warnings": warnings,
    }
    result["precontact_pass"] = (
        0.0005 <= final_gap <= 0.003
        and abs(float(result["face_angle_change_deg"])) <= 1.0
        and float(result["maximum_axis_drift_mm"]) < 0.3
        and float(result["thumb_middle_support_rate"]) == 1.0
        and float(result["ring_face_contact_rate"]) >= 0.95
        and not bool(result["penetration_over_0.002_m"])
        and not warnings
    )
    return result


def main() -> None:
    scene_before = sha256(SCENE)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    results = [evaluate(model, candidate) for candidate in load_candidates()]
    passing = sorted(
        (result for result in results if result["precontact_pass"]),
        key=lambda result: float(result["maximum_axis_drift_mm"]),
    )
    ranked = sorted(
        results,
        key=lambda result: (
            float(result["maximum_axis_drift_mm"]),
            abs(float(result["final_face_gap_m"]) - 0.0015),
        ),
    )
    if passing:
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "label": "articulated_index_face_precontact_R01",
                    "source": "KF03 endpoint",
                    "transition_steps": TRANSITION_STEPS,
                    "hold_steps": HOLD_STEPS,
                    "target_joints": passing[0]["targets"],
                    "measured": passing[0],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    lines = [
        "test=index-to-face dynamic pre-contact screen",
        f"candidate_count={len(results)}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_passing={json.dumps(passing[0], sort_keys=True) if passing else 'NONE'}",
        f"best_ranked={json.dumps(ranked[0], sort_keys=True) if ranked else 'NONE'}",
        f"INDEX_FACE_PRECONTACT={'PASS' if passing else 'FAIL'}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
