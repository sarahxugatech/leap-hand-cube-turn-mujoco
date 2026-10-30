"""Scan a DIP-first delay for the fixed Prompt 29 ring-finger keyframe."""

from __future__ import annotations

import csv
import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    RING_JOINTS,
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


OUTPUT_LOG = ROOT / "logs" / "prompt30_ring_phased_delay_search.txt"
OUTPUT_CSV = ROOT / "logs" / "prompt30_ring_phased_delay_traces.csv"
DELAYS = (30, 45, 60, 75)
DELTAS = {"rf_mcp": 0.08, "rf_pip": -0.06, "rf_dip": -0.10}
TRANSITION_STEPS = 150
HOLD_STEPS = 500


def cosine(step: int, total: int) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(step, total) / total)


def evaluate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    delay: int,
    starts: dict[str, float],
    baseline_angle: float,
    baseline_cube_xyz: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, float | int | str]]]:
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    hand_set = set(hand_geoms(model))
    actuator_ids = {name: actuator_id(model, name) for name in RING_JOINTS}
    targets = {name: starts[name] + DELTAS.get(name, 0.0) for name in RING_JOINTS}
    rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(TRANSITION_STEPS + HOLD_STEPS):
            if step < TRANSITION_STEPS:
                dip_blend = cosine(step + 1, TRANSITION_STEPS)
                push_blend = 0.0
                if step >= delay:
                    push_blend = cosine(step - delay + 1, TRANSITION_STEPS - delay)
            else:
                dip_blend = push_blend = 1.0
            commanded = dict(starts)
            commanded["rf_dip"] = starts["rf_dip"] + dip_blend * DELTAS["rf_dip"]
            for name in ("rf_mcp", "rf_pip"):
                commanded[name] = starts[name] + push_blend * DELTAS[name]
            for name, target in commanded.items():
                data.ctrl[actuator_ids[name]] = target
            mujoco.mj_step(model, data)

            rotation = data.xmat[face_body].reshape(3, 3)
            center = data.xpos[face_body]
            core_prefixes: set[str] = set()
            ring_core = False
            normal_force = 0.0
            face_contact = False
            minimum_distance = float("inf")
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
                            core_prefixes.add(prefix)
                        ring_core |= prefix == "rf"
                elif face_geom in pair:
                    other = second if first == face_geom else first
                    if other == rf_geom:
                        local = rotation.T @ (np.asarray(contact.pos) - center)
                        on_side = max(abs(local[0]), abs(local[1])) >= 0.023 and abs(local[2]) <= 0.0085
                        if on_side:
                            face_contact = True
                            force = np.zeros(6)
                            mujoco.mj_contactForce(model, data, contact_index, force)
                            normal_force += max(0.0, float(force[0]))
            drift = np.abs(data.xpos[cube_body] - baseline_cube_xyz)
            rows.append(
                {
                    "delay_steps": delay,
                    "step": step + 1,
                    "phase": "transition" if step < TRANSITION_STEPS else "hold",
                    "face_angle_deg": math.degrees(float(data.qpos[face_qpos])),
                    "normal_force_n": normal_force,
                    "face_contact": int(face_contact),
                    "core_support": int({"th", "if", "mf"}.issubset(core_prefixes)),
                    "ring_core": int(ring_core),
                    "minimum_hand_distance_m": minimum_distance,
                    "cube_abs_dx_mm": 1000 * float(drift[0]),
                    "cube_abs_dy_mm": 1000 * float(drift[1]),
                    "cube_abs_dz_mm": 1000 * float(drift[2]),
                    "rf_mcp_target_rad": commanded["rf_mcp"],
                    "rf_pip_target_rad": commanded["rf_pip"],
                    "rf_dip_target_rad": commanded["rf_dip"],
                }
            )
    finally:
        mujoco.set_mju_user_warning(old_handler)

    transition = rows[:TRANSITION_STEPS]
    hold = rows[TRANSITION_STEPS:]
    last_100 = rows[-100:]
    maximum_axis_drift_mm = max(
        max(float(row[key]) for key in ("cube_abs_dx_mm", "cube_abs_dy_mm", "cube_abs_dz_mm"))
        for row in rows
    )
    transition_peak_force = max(float(row["normal_force_n"]) for row in transition)
    minimum_distance = min(float(row["minimum_hand_distance_m"]) for row in rows)
    final_angle = float(rows[-1]["face_angle_deg"])
    result = {
        "delay_steps": delay,
        "delay_ms": 1000.0 * delay * float(model.opt.timestep),
        "final_angle_deg": final_angle,
        "angle_gain_deg": final_angle - math.degrees(baseline_angle),
        "last_100_range_deg": max(float(row["face_angle_deg"]) for row in last_100)
        - min(float(row["face_angle_deg"]) for row in last_100),
        "transition_peak_normal_force_n": transition_peak_force,
        "transition_steps_over_0.35_n": sum(float(row["normal_force_n"]) > 0.35 for row in transition),
        "transition_steps_over_0.5_n": sum(float(row["normal_force_n"]) > 0.5 for row in transition),
        "transition_contact_rate": sum(int(row["face_contact"]) for row in transition) / len(transition),
        "hold_contact_rate": sum(int(row["face_contact"]) for row in hold) / len(hold),
        "support_rate": sum(int(row["core_support"]) for row in rows) / len(rows),
        "ring_core_rate": sum(int(row["ring_core"]) for row in rows) / len(rows),
        "max_axis_drift_mm": maximum_axis_drift_mm,
        "max_xyz_drift_mm": [
            max(float(row[key]) for row in rows)
            for key in ("cube_abs_dx_mm", "cube_abs_dy_mm", "cube_abs_dz_mm")
        ],
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": minimum_distance < -PENETRATION_LIMIT,
        "warnings": warnings,
    }
    result["pass"] = (
        15.0 <= final_angle <= 20.0
        and maximum_axis_drift_mm <= 0.3
        and transition_peak_force <= 0.35
        and float(result["transition_contact_rate"]) >= 0.95
        and float(result["hold_contact_rate"]) >= 0.95
        and float(result["support_rate"]) == 1.0
        and float(result["ring_core_rate"]) == 0.0
        and float(result["last_100_range_deg"]) <= 0.25
        and not bool(result["penetration_over_0.002_m"])
        and not warnings
    )
    return result, rows


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before phased-delay search")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    starts = {name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS}
    baseline_angle = float(data.qpos[face_qpos])
    baseline_cube_xyz = data.xpos[cube_body].copy()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)

    results: list[dict[str, object]] = []
    all_rows: list[dict[str, float | int | str]] = []
    for delay in DELAYS:
        mujoco.mj_setState(model, data, state, state_spec)
        mujoco.mj_forward(model, data)
        result, rows = evaluate(model, data, delay, starts, baseline_angle, baseline_cube_xyz)
        results.append(result)
        all_rows.extend(rows)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during phased-delay search")
    passing = sorted(
        (result for result in results if result["pass"]),
        key=lambda result: float(result["max_axis_drift_mm"]),
    )
    lines = [
        "test=Prompt 30 DIP-first phased-delay scan",
        f"dt_s={float(model.opt.timestep)}",
        f"fixed_deltas_rad={json.dumps(DELTAS, sort_keys=True)}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_passing={json.dumps(passing[0], sort_keys=True) if passing else 'NONE'}",
        f"trace={OUTPUT_CSV}",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
