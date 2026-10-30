"""Scan MCP compression and positive ROT for ring re-engagement."""

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


OUTPUT_LOG = ROOT / "logs" / "prompt33_reengagement_geometry_search.txt"
OUTPUT_CSV = ROOT / "logs" / "prompt33_reengagement_geometry_traces.csv"
MCP_DELTAS = (0.10, 0.14, 0.18, 0.22)
ROT_DELTAS = (0.0, 0.03, 0.06, 0.09)
DIP_DELTA = 0.04
PIP_DELTA = -0.06
DELAY = 30
TRANSITION_STEPS = 150
HOLD_STEPS = 500
FORCE_CONTACT_N = 0.001


def cosine(step: int, total: int) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(step, total) / total)


def evaluate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mcp_delta: float,
    rot_delta: float,
    starts: dict[str, float],
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
    rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(TRANSITION_STEPS + HOLD_STEPS):
            if step < TRANSITION_STEPS:
                dip_blend = cosine(step + 1, TRANSITION_STEPS)
                push_blend = 0.0 if step < DELAY else cosine(
                    step - DELAY + 1, TRANSITION_STEPS - DELAY
                )
            else:
                dip_blend = push_blend = 1.0
            commanded = dict(starts)
            commanded["rf_dip"] = starts["rf_dip"] + dip_blend * DIP_DELTA
            commanded["rf_mcp"] = starts["rf_mcp"] + push_blend * mcp_delta
            commanded["rf_pip"] = starts["rf_pip"] + push_blend * PIP_DELTA
            commanded["rf_rot"] = starts["rf_rot"] + push_blend * rot_delta
            for name, value in commanded.items():
                data.ctrl[actuator_ids[name]] = value
            mujoco.mj_step(model, data)

            rotation = data.xmat[face_body].reshape(3, 3)
            center = data.xpos[face_body]
            normal_force = 0.0
            core_prefixes: set[str] = set()
            ring_core = False
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
                            force = np.zeros(6)
                            mujoco.mj_contactForce(model, data, contact_index, force)
                            normal_force += max(0.0, float(force[0]))
            drift = np.abs(data.xpos[cube_body] - baseline_cube_xyz)
            rows.append(
                {
                    "mcp_delta_rad": mcp_delta,
                    "rot_delta_rad": rot_delta,
                    "step": step + 1,
                    "phase": "transition" if step < TRANSITION_STEPS else "hold",
                    "face_angle_deg": math.degrees(float(data.qpos[face_qpos])),
                    "normal_force_n": normal_force,
                    "force_contact": int(normal_force > FORCE_CONTACT_N),
                    "core_support": int({"th", "if", "mf"}.issubset(core_prefixes)),
                    "ring_core": int(ring_core),
                    "minimum_hand_distance_m": minimum_distance,
                    "cube_abs_dx_mm": 1000 * float(drift[0]),
                    "cube_abs_dy_mm": 1000 * float(drift[1]),
                    "cube_abs_dz_mm": 1000 * float(drift[2]),
                }
            )
    finally:
        mujoco.set_mju_user_warning(old_handler)

    transition = rows[:TRANSITION_STEPS]
    hold = rows[TRANSITION_STEPS:]
    last_100 = rows[-100:]
    transition_max_xyz = [
        max(float(row[key]) for row in transition)
        for key in ("cube_abs_dx_mm", "cube_abs_dy_mm", "cube_abs_dz_mm")
    ]
    full_max_xyz = [
        max(float(row[key]) for row in rows)
        for key in ("cube_abs_dx_mm", "cube_abs_dy_mm", "cube_abs_dz_mm")
    ]
    minimum_distance = min(float(row["minimum_hand_distance_m"]) for row in rows)
    result = {
        "mcp_delta_rad": mcp_delta,
        "rot_delta_rad": rot_delta,
        "transition_end_angle_deg": float(transition[-1]["face_angle_deg"]),
        "hold_final_angle_deg": float(rows[-1]["face_angle_deg"]),
        "transition_contact_rate": sum(int(row["force_contact"]) for row in transition)
        / len(transition),
        "hold_contact_rate": sum(int(row["force_contact"]) for row in hold) / len(hold),
        "last_100_contact_rate": sum(int(row["force_contact"]) for row in last_100) / 100,
        "transition_peak_normal_force_n": max(float(row["normal_force_n"]) for row in transition),
        "transition_max_xyz_drift_mm": transition_max_xyz,
        "transition_max_axis_drift_mm": max(transition_max_xyz),
        "full_max_xyz_drift_mm": full_max_xyz,
        "full_max_axis_drift_mm": max(full_max_xyz),
        "support_rate": sum(int(row["core_support"]) for row in rows) / len(rows),
        "ring_core_rate": sum(int(row["ring_core"]) for row in rows) / len(rows),
        "last_100_angle_range_deg": max(float(row["face_angle_deg"]) for row in last_100)
        - min(float(row["face_angle_deg"]) for row in last_100),
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": minimum_distance < -PENETRATION_LIMIT,
        "warnings": warnings,
    }
    result["redlines_pass"] = (
        float(result["transition_contact_rate"]) >= 0.85
        and float(result["transition_end_angle_deg"]) > 13.5
        and float(result["transition_max_axis_drift_mm"]) < 0.4
    )
    result["project_pass"] = (
        bool(result["redlines_pass"])
        and float(result["hold_final_angle_deg"]) > 13.5
        and float(result["last_100_contact_rate"]) >= 0.95
        and float(result["full_max_axis_drift_mm"]) < 0.4
        and float(result["support_rate"]) == 1.0
        and float(result["ring_core_rate"]) == 0.0
        and float(result["last_100_angle_range_deg"]) <= 0.25
        and not bool(result["penetration_over_0.002_m"])
        and not warnings
    )
    return result, rows


def initial_state(
    model: mujoco.MjModel,
) -> tuple[mujoco.MjData, dict[str, float], np.ndarray, np.ndarray, mujoco.mjtState]:
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    starts = {name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS}
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    baseline_cube_xyz = data.xpos[cube_body].copy()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)
    return data, starts, baseline_cube_xyz, state, state_spec


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before geometry scan")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data, starts, baseline_cube_xyz, state, state_spec = initial_state(model)
    results: list[dict[str, object]] = []
    all_rows: list[dict[str, float | int | str]] = []
    for mcp_delta in MCP_DELTAS:
        for rot_delta in ROT_DELTAS:
            mujoco.mj_setState(model, data, state, state_spec)
            mujoco.mj_forward(model, data)
            result, rows = evaluate(
                model, data, mcp_delta, rot_delta, starts, baseline_cube_xyz
            )
            results.append(result)
            all_rows.extend(rows)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    redline_passing = sorted(
        (result for result in results if result["redlines_pass"]),
        key=lambda result: (
            -float(result["transition_end_angle_deg"]),
            float(result["transition_max_axis_drift_mm"]),
        ),
    )
    project_passing = [result for result in redline_passing if result["project_pass"]]
    repeat_runs: list[dict[str, object]] = []
    if project_passing:
        best = project_passing[0]
        for _ in range(2):
            clean_data, clean_starts, clean_cube_xyz, _, _ = initial_state(model)
            replay, _ = evaluate(
                model,
                clean_data,
                float(best["mcp_delta_rad"]),
                float(best["rot_delta_rad"]),
                clean_starts,
                clean_cube_xyz,
            )
            repeat_runs.append(replay)

    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during geometry scan")
    lines = [
        "test=Prompt 33 reengagement geometry scan",
        f"fixed_dip_delta_rad={DIP_DELTA}",
        f"fixed_pip_delta_rad={PIP_DELTA}",
        f"delay_steps={DELAY}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_redlines={json.dumps(redline_passing[0], sort_keys=True) if redline_passing else 'NONE'}",
        f"best_project={json.dumps(project_passing[0], sort_keys=True) if project_passing else 'NONE'}",
        f"repeat_runs={json.dumps(repeat_runs, sort_keys=True)}",
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
