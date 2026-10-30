"""Validate the new true-ring-tip topology through hold and lift."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import MAIN_SCENE, SCENE
from search_global_natural_ring_tip_grasp import OUTPUT_LOG as PLANNER_LOG
from search_global_redundant_grasp import EXPECTED_MAIN_SHA256, EXPECTED_SCENE_SHA256
from search_natural_redundant_grasp import set_candidate
from validate_poses import (
    PENETRATION_LIMIT,
    ROOT,
    actuator_id,
    finger_prefix,
    geom_name,
    hand_geoms,
    object_id,
    sha256,
)


OUTPUT_LOG = ROOT / "logs" / "prompt41_global_natural_ring_tip_grasp_validation.txt"
OUTPUT_POSE = ROOT / "poses" / "global_natural_ring_tip_grasp_R01.json"
HOLD_STEPS = 500
LIFT_STEPS = 300
LIFT_HOLD_STEPS = 500
MAX_HOLD_AXIS_DRIFT_MM = 0.5
MAX_LIFT_RELATIVE_AXIS_DRIFT_MM = 0.6


def candidates() -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("candidate="))
        for line in PLANNER_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("candidate=") and json.loads(line.removeprefix("candidate="))["static_gate"]
    ]


def evaluate(model: mujoco.MjModel, candidate: dict[str, object]) -> dict[str, object]:
    data = mujoco.MjData(model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    tip_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_tip")
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    hand_set = set(hand_geoms(model))
    set_candidate(
        model,
        data,
        cube_qpos,
        face_qpos,
        candidate["cube_xyz"],
        candidate["cube_quat_wxyz"],
        candidate["mount"],
        candidate["joints"],
    )
    start_cube = data.xpos[cube_body].copy()
    start_relative = data.xpos[cube_body] - data.xpos[palm_body]
    start_face = float(data.qpos[face_qpos])
    maximum_hold_drift = np.zeros(3)
    maximum_lift_relative_drift = np.zeros(3)
    tip_face_contacts: list[bool] = []
    tip_face_forces: list[float] = []
    support: list[bool] = []
    minimum_distance = float("inf")
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        total = HOLD_STEPS + LIFT_STEPS + LIFT_HOLD_STEPS
        for step in range(total):
            if step >= HOLD_STEPS:
                lift_step = min(step - HOLD_STEPS + 1, LIFT_STEPS)
                blend = 0.5 - 0.5 * math.cos(math.pi * lift_step / LIFT_STEPS)
                data.ctrl[actuator_id(model, "mount_z")] = (
                    float(candidate["mount"]["mount_z"]) + 0.030 * blend
                )
            mujoco.mj_step(model, data)
            prefixes: set[str] = set()
            tip_force = 0.0
            tip_contact = False
            for index, contact in enumerate(data.contact[: data.ncon]):
                first, second = int(contact.geom1), int(contact.geom2)
                if first in hand_set or second in hand_set:
                    minimum_distance = min(minimum_distance, float(contact.dist))
                pair = {first, second}
                if core_geom in pair:
                    other = second if first == core_geom else first
                    if other in hand_set:
                        prefix = finger_prefix(geom_name(model, other))
                        if prefix:
                            prefixes.add(prefix)
                if face_geom in pair:
                    other = second if first == face_geom else first
                    if other == tip_geom:
                        tip_contact = True
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, index, force)
                        tip_force += max(0.0, float(force[0]))
            tip_face_contacts.append(tip_contact)
            tip_face_forces.append(tip_force)
            support.append({"th", "if", "mf"}.issubset(prefixes))
            if step < HOLD_STEPS:
                maximum_hold_drift = np.maximum(
                    maximum_hold_drift, np.abs(data.xpos[cube_body] - start_cube)
                )
            else:
                relative = data.xpos[cube_body] - data.xpos[palm_body]
                maximum_lift_relative_drift = np.maximum(
                    maximum_lift_relative_drift, np.abs(relative - start_relative)
                )
    finally:
        mujoco.set_mju_user_warning(old_handler)

    result: dict[str, object] = {
        "sample": candidate["sample"],
        "tip_face_last100_contact_rate": sum(tip_face_contacts[-100:]) / 100,
        "tip_face_last100_average_force_n": float(np.mean(tip_face_forces[-100:])),
        "tip_face_peak_force_n": max(tip_face_forces),
        "support_last100_rate": sum(support[-100:]) / 100,
        "support_full_rate": sum(support) / len(support),
        "maximum_hold_axis_drift_mm": 1000 * float(np.max(maximum_hold_drift)),
        "maximum_lift_relative_axis_drift_mm": 1000
        * float(np.max(maximum_lift_relative_drift)),
        "face_angle_change_deg": math.degrees(float(data.qpos[face_qpos]) - start_face),
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": minimum_distance < -PENETRATION_LIMIT,
        "warnings": warnings,
        "candidate": candidate,
    }
    result["hold_lift_pass"] = (
        result["tip_face_last100_contact_rate"] >= 0.95
        and 0.001 <= result["tip_face_last100_average_force_n"] <= 0.35
        and result["support_last100_rate"] >= 0.95
        and result["maximum_hold_axis_drift_mm"] <= MAX_HOLD_AXIS_DRIFT_MM
        and result["maximum_lift_relative_axis_drift_mm"] <= MAX_LIFT_RELATIVE_AXIS_DRIFT_MM
        and abs(result["face_angle_change_deg"]) <= 2.0
        and not result["penetration_over_0.002_m"]
        and not warnings
    )
    return result


def main() -> None:
    if sha256(MAIN_SCENE) != EXPECTED_MAIN_SHA256 or sha256(SCENE) != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    results = [evaluate(model, candidate) for candidate in candidates()]
    passing = [result for result in results if result["hold_lift_pass"]]
    if passing:
        best = min(passing, key=lambda result: float(result["maximum_hold_axis_drift_mm"]))
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "test_only": True,
                    "label": "global_natural_ring_tip_grasp_R01",
                    "candidate": best["candidate"],
                    "measured": {key: value for key, value in best.items() if key != "candidate"},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    lines = [
        "test=global natural true-ring-tip grasp hold/lift validation",
        f"candidate_count={len(results)}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_passing={json.dumps(passing[0], sort_keys=True) if passing else 'NONE'}",
        f"NATURAL_RING_TIP_GRASP={'PASS' if passing else 'FAIL'}",
        f"promoted_pose={OUTPUT_POSE if passing else 'NONE'}",
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
