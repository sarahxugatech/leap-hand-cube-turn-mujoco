"""Dynamically validate topologies from search_global_redundant_grasp.py."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import MAIN_SCENE, SCENE
from search_global_redundant_grasp import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    OUTPUT_LOG as PLANNER_LOG,
)
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


OUTPUT_LOG = ROOT / "logs" / "prompt39_global_redundant_grasp_validation.txt"
OUTPUT_POSE = ROOT / "poses" / "global_redundant_grasp_R01.json"
HOLD_STEPS = 500
LIFT_STEPS = 300
LIFT_HOLD_STEPS = 500
FORCE_MIN_N = 0.01
FORCE_MAX_N = 0.15
STRICT_HOLD_DRIFT_MM = 0.1
STRICT_LIFT_RELATIVE_DRIFT_MM = 0.1


def candidates() -> list[dict[str, object]]:
    items = [
        json.loads(line.removeprefix("candidate="))
        for line in PLANNER_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("candidate=")
    ]
    passing = [item for item in items if item["static_gate"]]
    return (passing or items)[:20]


def contact_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    core_geom: int,
    face_geom: int,
    ring_tip: int,
    hand_set: set[int],
) -> tuple[float, float, set[str], float]:
    ring_core_force = 0.0
    ring_face_force = 0.0
    core_prefixes: set[str] = set()
    minimum_distance = float("inf")
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
                    core_prefixes.add(prefix)
            if other == ring_tip:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                ring_core_force += max(0.0, float(force[0]))
        if face_geom in pair:
            other = second if first == face_geom else first
            if other == ring_tip:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                ring_face_force += max(0.0, float(force[0]))
    return ring_core_force, ring_face_force, core_prefixes, minimum_distance


def evaluate(model: mujoco.MjModel, candidate: dict[str, object]) -> dict[str, object]:
    data = mujoco.MjData(model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    ring_tip = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    hand_set = set(hand_geoms(model))
    data.qpos[cube_qpos : cube_qpos + 7] = (
        *candidate["cube_xyz"],
        *candidate["cube_quat_wxyz"],
    )
    data.qpos[face_qpos] = 0.0
    for name, value in candidate["mount"].items() | candidate["joints"].items():
        data.qpos[qpos_address(model, name)] = float(value)
        data.ctrl[actuator_id(model, name)] = float(value)
    mujoco.mj_forward(model, data)
    start_cube = data.xpos[cube_body].copy()
    start_relative = data.xpos[cube_body] - data.xpos[palm_body]
    start_face = float(data.qpos[face_qpos])
    ring_core_forces: list[float] = []
    ring_face_forces: list[float] = []
    support: list[bool] = []
    maximum_hold_drift = np.zeros(3)
    maximum_lift_relative_drift = np.zeros(3)
    minimum_distance = float("inf")
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(HOLD_STEPS + LIFT_STEPS + LIFT_HOLD_STEPS):
            if step >= HOLD_STEPS:
                lift_step = min(step - HOLD_STEPS + 1, LIFT_STEPS)
                blend = 0.5 - 0.5 * math.cos(math.pi * lift_step / LIFT_STEPS)
                data.ctrl[actuator_id(model, "mount_z")] = (
                    float(candidate["mount"]["mount_z"]) + 0.030 * blend
                )
            mujoco.mj_step(model, data)
            core_force, face_force, prefixes, distance = contact_metrics(
                model, data, core_geom, face_geom, ring_tip, hand_set
            )
            ring_core_forces.append(core_force)
            ring_face_forces.append(face_force)
            support.append({"th", "if", "mf"}.issubset(prefixes))
            minimum_distance = min(minimum_distance, distance)
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

    def healthy_last100(values: list[float]) -> float:
        return sum(FORCE_MIN_N <= value <= FORCE_MAX_N for value in values[-100:]) / 100

    result: dict[str, object] = {
        "sample": candidate["sample"],
        "static_gate": candidate["static_gate"],
        "ring_core_last100_healthy_rate": healthy_last100(ring_core_forces),
        "ring_face_last100_healthy_rate": healthy_last100(ring_face_forces),
        "ring_core_peak_force_n": max(ring_core_forces),
        "ring_face_peak_force_n": max(ring_face_forces),
        "thumb_index_middle_support_rate": sum(support) / len(support),
        "thumb_index_middle_last100_rate": sum(support[-100:]) / 100,
        "maximum_hold_xyz_drift_mm": [1000 * float(value) for value in maximum_hold_drift],
        "maximum_hold_axis_drift_mm": 1000 * float(np.max(maximum_hold_drift)),
        "maximum_lift_relative_xyz_drift_mm": [
            1000 * float(value) for value in maximum_lift_relative_drift
        ],
        "maximum_lift_relative_axis_drift_mm": 1000
        * float(np.max(maximum_lift_relative_drift)),
        "face_angle_change_deg": math.degrees(float(data.qpos[face_qpos]) - start_face),
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": minimum_distance < -PENETRATION_LIMIT,
        "warnings": warnings,
        "candidate": candidate,
    }
    result["hold_lift_pass"] = (
        result["ring_core_last100_healthy_rate"] >= 0.95
        and result["ring_face_last100_healthy_rate"] >= 0.95
        and result["thumb_index_middle_last100_rate"] >= 0.95
        and result["maximum_hold_axis_drift_mm"] <= STRICT_HOLD_DRIFT_MM
        and result["maximum_lift_relative_axis_drift_mm"] <= STRICT_LIFT_RELATIVE_DRIFT_MM
        and not result["penetration_over_0.002_m"]
        and not warnings
    )
    return result


def main() -> None:
    if sha256(MAIN_SCENE) != EXPECTED_MAIN_SHA256 or sha256(SCENE) != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before dynamic validation")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    results = [evaluate(model, candidate) for candidate in candidates()]
    passing = [result for result in results if result["hold_lift_pass"]]
    ranked = sorted(
        results,
        key=lambda result: (
            -min(
                float(result["ring_core_last100_healthy_rate"]),
                float(result["ring_face_last100_healthy_rate"]),
            ),
            float(result["maximum_hold_axis_drift_mm"]),
        ),
    )
    if passing:
        best = passing[0]
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "test_only": True,
                    "label": "global_redundant_grasp_R01",
                    "source": str(PLANNER_LOG),
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
        "test=global redundant-grasp dynamic hold and lift validation",
        f"candidate_count={len(results)}",
        f"hold_steps={HOLD_STEPS}",
        f"lift_steps={LIFT_STEPS}",
        f"lift_hold_steps={LIFT_HOLD_STEPS}",
        f"strict_hold_axis_drift_mm={STRICT_HOLD_DRIFT_MM}",
        f"strict_lift_relative_axis_drift_mm={STRICT_LIFT_RELATIVE_DRIFT_MM}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_passing={json.dumps(passing[0], sort_keys=True) if passing else 'NONE'}",
        f"best_ranked={json.dumps(ranked[0], sort_keys=True) if ranked else 'NONE'}",
        f"GLOBAL_REDUNDANT_GRASP={'PASS' if passing else 'FAIL'}",
        "release_test=NOT_RUN (requires hold+lift pass)",
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7]))
    print(lines[-6])
    print(lines[-5])
    print(lines[-4])


if __name__ == "__main__":
    main()
