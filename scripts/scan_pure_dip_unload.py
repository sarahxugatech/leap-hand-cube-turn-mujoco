"""Scan pure ring-DIP motion from the validated KF02 state."""

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


OUTPUT_LOG = ROOT / "logs" / "prompt31_pure_dip_unload_scan.txt"
OUTPUT_CSV = ROOT / "logs" / "prompt31_pure_dip_unload_traces.csv"
DELTAS = (-0.12, -0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12)
SCAN_STEPS = 60


def evaluate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    delta: float,
    starts: dict[str, float],
    baseline_cube_xyz: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, float | int]]]:
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    hand_set = set(hand_geoms(model))
    actuator_ids = {name: actuator_id(model, name) for name in RING_JOINTS}
    rows: list[dict[str, float | int]] = []
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(SCAN_STEPS):
            blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / SCAN_STEPS)
            commanded = dict(starts)
            commanded["rf_dip"] = starts["rf_dip"] + blend * delta
            for name, value in commanded.items():
                data.ctrl[actuator_ids[name]] = value
            mujoco.mj_step(model, data)

            rotation = data.xmat[face_body].reshape(3, 3)
            center = data.xpos[face_body]
            normal_force = 0.0
            face_contact = False
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
                            face_contact = True
                            force = np.zeros(6)
                            mujoco.mj_contactForce(model, data, contact_index, force)
                            normal_force += max(0.0, float(force[0]))
            drift = np.abs(data.xpos[cube_body] - baseline_cube_xyz)
            rows.append(
                {
                    "dip_delta_rad": delta,
                    "step": step + 1,
                    "dip_target_rad": commanded["rf_dip"],
                    "dip_actual_rad": float(data.qpos[qpos_address(model, "rf_dip")]),
                    "face_angle_deg": math.degrees(float(data.qpos[face_qpos])),
                    "normal_force_n": normal_force,
                    "face_contact": int(face_contact),
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

    peak_force = max(float(row["normal_force_n"]) for row in rows)
    max_xyz = [
        max(float(row[key]) for row in rows)
        for key in ("cube_abs_dx_mm", "cube_abs_dy_mm", "cube_abs_dz_mm")
    ]
    max_drift = max(max_xyz)
    minimum_distance = min(float(row["minimum_hand_distance_m"]) for row in rows)
    final_angle = float(rows[-1]["face_angle_deg"])
    result = {
        "dip_delta_rad": delta,
        "peak_normal_force_n": peak_force,
        "final_normal_force_n": float(rows[-1]["normal_force_n"]),
        "final_face_angle_deg": final_angle,
        "maximum_axis_drift_mm": max_drift,
        "maximum_xyz_drift_mm": max_xyz,
        "contact_rate": sum(int(row["face_contact"]) for row in rows) / len(rows),
        "support_rate": sum(int(row["core_support"]) for row in rows) / len(rows),
        "ring_core_rate": sum(int(row["ring_core"]) for row in rows) / len(rows),
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": minimum_distance < -PENETRATION_LIMIT,
        "warnings": warnings,
    }
    result["eligible"] = (
        final_angle >= 9.3
        and max_drift < 0.1
        and float(result["support_rate"]) == 1.0
        and float(result["ring_core_rate"]) == 0.0
        and not bool(result["penetration_over_0.002_m"])
        and not warnings
    )
    return result, rows


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before pure-DIP scan")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    starts = {name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS}
    baseline_cube_xyz = data.xpos[cube_body].copy()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)

    results: list[dict[str, object]] = []
    all_rows: list[dict[str, float | int]] = []
    for delta in DELTAS:
        mujoco.mj_setState(model, data, state, state_spec)
        mujoco.mj_forward(model, data)
        result, rows = evaluate(model, data, delta, starts, baseline_cube_xyz)
        results.append(result)
        all_rows.extend(rows)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    eligible = sorted(
        (result for result in results if result["eligible"]),
        key=lambda result: float(result["peak_normal_force_n"]),
    )
    force_ranked = sorted(results, key=lambda result: float(result["peak_normal_force_n"]))
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during pure-DIP scan")
    lines = [
        "test=Prompt 31 pure DIP unload direction/amplitude scan",
        f"scan_steps={SCAN_STEPS}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_eligible={json.dumps(eligible[0], sort_keys=True) if eligible else 'NONE'}",
        f"lowest_peak_force_any={json.dumps(force_ranked[0], sort_keys=True)}",
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
