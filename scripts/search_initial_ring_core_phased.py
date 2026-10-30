"""Search phased joint-order paths to the planned initial ring-core poses."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from plan_initial_ring_core_support import OUTPUT_LOG as PLANNER_LOG, replay_lifted_grasp
from search_articulated_face_followthrough import RING_JOINTS, SCENE
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


OUTPUT_LOG = ROOT / "logs" / "prompt38_initial_ring_core_phased_search.txt"
OUTPUT_POSE = ROOT / "poses" / "initial_ring_core_support_phased_R01.json"
PHASE_STEPS = 150
HOLD_STEPS = 300
SCHEDULES = {
    "rot_then_curl": (("rf_rot",), ("rf_mcp", "rf_pip", "rf_dip")),
    "curl_then_rot": (("rf_mcp", "rf_pip", "rf_dip"), ("rf_rot",)),
    "distal_then_base": (("rf_pip", "rf_dip"), ("rf_mcp", "rf_rot")),
    "base_then_distal": (("rf_mcp", "rf_rot"), ("rf_pip", "rf_dip")),
    "rot_dip_then_mcp_pip": (("rf_rot", "rf_dip"), ("rf_mcp", "rf_pip")),
    "mcp_pip_then_rot_dip": (("rf_mcp", "rf_pip"), ("rf_rot", "rf_dip")),
}


def candidates() -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("candidate="))
        for line in PLANNER_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("candidate=")
    ][:10]


def evaluate(
    model: mujoco.MjModel,
    candidate: dict[str, object],
    schedule_name: str,
) -> dict[str, object]:
    data = mujoco.MjData(model)
    replay_lifted_grasp(model, data)
    starts = {name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS}
    targets = {name: float(value) for name, value in candidate["targets"].items()}
    cube = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    tip_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    hand_set = set(hand_geoms(model))
    baseline_relative = data.xpos[cube] - data.xpos[palm]
    baseline_angle = float(data.qpos[face_qpos])
    commands = starts.copy()
    maximum_drift = np.zeros(3)
    minimum_distance = float("inf")
    base_support = 0
    ring_core: list[bool] = []
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for phase_joints in SCHEDULES[schedule_name]:
            phase_start = commands.copy()
            for step in range(PHASE_STEPS):
                blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / PHASE_STEPS)
                for name in RING_JOINTS:
                    target = targets[name] if name in phase_joints else phase_start[name]
                    data.ctrl[actuator_id(model, name)] = phase_start[name] + blend * (
                        target - phase_start[name]
                    )
                mujoco.mj_step(model, data)
                commands.update(
                    {name: targets[name] for name in phase_joints}
                    if step == PHASE_STEPS - 1
                    else {}
                )
                prefixes, contact, distance = observe(
                    model, data, core_geom, tip_geom, hand_set
                )
                base_support += {"th", "if", "mf"}.issubset(prefixes)
                ring_core.append(contact)
                minimum_distance = min(minimum_distance, distance)
                relative = data.xpos[cube] - data.xpos[palm]
                maximum_drift = np.maximum(
                    maximum_drift, np.abs(relative - baseline_relative)
                )
        for _ in range(HOLD_STEPS):
            mujoco.mj_step(model, data)
            prefixes, contact, distance = observe(
                model, data, core_geom, tip_geom, hand_set
            )
            base_support += {"th", "if", "mf"}.issubset(prefixes)
            ring_core.append(contact)
            minimum_distance = min(minimum_distance, distance)
            relative = data.xpos[cube] - data.xpos[palm]
            maximum_drift = np.maximum(maximum_drift, np.abs(relative - baseline_relative))
    finally:
        mujoco.set_mju_user_warning(old_handler)
    total = 2 * PHASE_STEPS + HOLD_STEPS
    result = {
        "schedule": schedule_name,
        "targets": targets,
        "face_angle_change_deg": math.degrees(float(data.qpos[face_qpos]) - baseline_angle),
        "maximum_axis_drift_mm": 1000 * float(np.max(maximum_drift)),
        "thumb_index_middle_support_rate": base_support / total,
        "ring_core_contact_rate": sum(ring_core) / total,
        "ring_core_last100_rate": sum(ring_core[-100:]) / 100,
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": minimum_distance < -PENETRATION_LIMIT,
        "warnings": warnings,
    }
    result["pass"] = (
        abs(float(result["face_angle_change_deg"])) <= 0.2
        and float(result["maximum_axis_drift_mm"]) < 0.3
        and float(result["thumb_index_middle_support_rate"]) == 1.0
        and float(result["ring_core_last100_rate"]) >= 0.95
        and not bool(result["penetration_over_0.002_m"])
        and not warnings
    )
    return result


def observe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    core_geom: int,
    tip_geom: int,
    hand_set: set[int],
) -> tuple[set[str], bool, float]:
    prefixes: set[str] = set()
    ring_contact = False
    minimum_distance = float("inf")
    for contact in data.contact[: data.ncon]:
        first, second = int(contact.geom1), int(contact.geom2)
        if first in hand_set or second in hand_set:
            minimum_distance = min(minimum_distance, float(contact.dist))
        if core_geom in {first, second}:
            other = second if first == core_geom else first
            if other in hand_set:
                prefix = finger_prefix(geom_name(model, other))
                if prefix:
                    prefixes.add(prefix)
            ring_contact |= other == tip_geom
    return prefixes, ring_contact, minimum_distance


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    results = [
        evaluate(model, candidate, schedule)
        for candidate in candidates()
        for schedule in SCHEDULES
    ]
    passing = sorted(
        (result for result in results if result["pass"]),
        key=lambda result: float(result["maximum_axis_drift_mm"]),
    )
    ranked = sorted(
        results,
        key=lambda result: (
            -float(result["ring_core_last100_rate"]),
            float(result["maximum_axis_drift_mm"]),
        ),
    )
    if passing:
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "label": "initial_ring_core_support_phased_R01",
                    "source": "validated lifted grasp",
                    "phase_steps": PHASE_STEPS,
                    "hold_steps": HOLD_STEPS,
                    "schedule": passing[0]["schedule"],
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
        "test=initial ring-core phased-path search",
        f"candidate_count={len(candidates())}",
        f"schedule_count={len(SCHEDULES)}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_passing={json.dumps(passing[0], sort_keys=True) if passing else 'NONE'}",
        f"best_ranked={json.dumps(ranked[0], sort_keys=True)}",
        f"PHASED_RING_CORE_SUPPORT={'PASS' if passing else 'FAIL'}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
