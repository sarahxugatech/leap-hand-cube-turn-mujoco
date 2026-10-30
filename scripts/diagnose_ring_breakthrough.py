"""Diagnose one large-step ring-finger follow-through from validated KF02."""

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
    ROOT,
    actuator_id,
    object_id,
    qpos_address,
    sha256,
)


OUTPUT_CSV = ROOT / "logs" / "prompt29_ring_breakthrough_trace.csv"
OUTPUT_LOG = ROOT / "logs" / "prompt29_ring_breakthrough_diagnostic.txt"
SEARCH_STEPS = 150
HOLD_STEPS = 500
DELTAS = {"rf_mcp": 0.08, "rf_pip": -0.06, "rf_dip": -0.10}
CONTACT_THRESHOLD_N = 0.005


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before breakthrough diagnostic")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)

    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    baseline_cube_xyz = data.xpos[cube_body].copy()
    baseline_angle = float(data.qpos[face_qpos])

    starts = {name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS}
    targets = dict(starts)
    for name, delta in DELTAS.items():
        targets[name] += delta
    actuator_ids = {name: actuator_id(model, name) for name in RING_JOINTS}
    force_limits = {
        name: {
            "limited": bool(model.actuator_forcelimited[aid]),
            "range": [float(value) for value in model.actuator_forcerange[aid]],
        }
        for name, aid in actuator_ids.items()
    }

    for name, target in targets.items():
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = map(float, model.jnt_range[joint])
        if not lower <= target <= upper:
            raise RuntimeError(f"{name} target {target} outside [{lower}, {upper}]")

    rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(SEARCH_STEPS + HOLD_STEPS):
            blend = 1.0
            if step < SEARCH_STEPS:
                blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / SEARCH_STEPS)
            commanded = {
                name: starts[name] + blend * (targets[name] - starts[name])
                for name in RING_JOINTS
            }
            for name, target in commanded.items():
                data.ctrl[actuator_ids[name]] = target
            mujoco.mj_step(model, data)

            normal_force = 0.0
            minimum_distance = float("inf")
            face_contact = False
            for contact_index, contact in enumerate(data.contact[: data.ncon]):
                pair = {int(contact.geom1), int(contact.geom2)}
                if rf_geom not in pair:
                    continue
                minimum_distance = min(minimum_distance, float(contact.dist))
                if face_geom in pair:
                    face_contact = True
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    normal_force += max(0.0, float(force[0]))

            row: dict[str, float | int | str] = {
                "step": step + 1,
                "phase": "transition" if step < SEARCH_STEPS else "hold",
                "face_angle_deg": math.degrees(float(data.qpos[face_qpos])),
                "normal_force_n": normal_force,
                "face_contact": int(face_contact),
                "rf_min_dist_mm": 1000.0 * minimum_distance
                if math.isfinite(minimum_distance)
                else float("nan"),
                "cube_dx_mm": 1000.0 * float(data.xpos[cube_body, 0] - baseline_cube_xyz[0]),
                "cube_dy_mm": 1000.0 * float(data.xpos[cube_body, 1] - baseline_cube_xyz[1]),
                "cube_dz_mm": 1000.0 * float(data.xpos[cube_body, 2] - baseline_cube_xyz[2]),
            }
            for name in RING_JOINTS:
                actual = float(data.qpos[qpos_address(model, name)])
                row[f"{name}_target_rad"] = commanded[name]
                row[f"{name}_actual_rad"] = actual
                row[f"{name}_error_rad"] = commanded[name] - actual
                row[f"{name}_actuator_force"] = float(data.actuator_force[actuator_ids[name]])
            rows.append(row)
    finally:
        mujoco.set_mju_user_warning(old_handler)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    transition = rows[:SEARCH_STEPS]
    hold = rows[SEARCH_STEPS:]
    maximum_force = max(float(row["normal_force_n"]) for row in rows)
    force_peak_step = max(rows, key=lambda row: float(row["normal_force_n"]))["step"]
    contact_rate = sum(int(row["face_contact"]) for row in rows) / len(rows)
    transition_contact_rate = sum(int(row["face_contact"]) for row in transition) / len(transition)
    hold_contact_rate = sum(int(row["face_contact"]) for row in hold) / len(hold)
    first_contact_loss = next(
        (int(row["step"]) for row in rows if not int(row["face_contact"])), None
    )
    min_dist_mm = min(
        float(row["rf_min_dist_mm"])
        for row in rows
        if math.isfinite(float(row["rf_min_dist_mm"]))
    )
    max_errors = {
        name: max(abs(float(row[f"{name}_error_rad"])) for row in rows)
        for name in RING_JOINTS
    }
    final_errors = {
        name: float(rows[-1][f"{name}_error_rad"]) for name in RING_JOINTS
    }
    max_actuator_forces = {
        name: max(abs(float(row[f"{name}_actuator_force"])) for row in rows)
        for name in RING_JOINTS
    }
    final_angle = float(rows[-1]["face_angle_deg"])
    peak_angle = max(float(row["face_angle_deg"]) for row in rows)
    final_gain = final_angle - math.degrees(baseline_angle)
    peak_gain = peak_angle - math.degrees(baseline_angle)
    max_axis_drift = max(
        max(abs(float(row[key])) for key in ("cube_dx_mm", "cube_dy_mm", "cube_dz_mm"))
        for row in rows
    )
    summary = {
        "deltas_rad": DELTAS,
        "baseline_angle_deg": math.degrees(baseline_angle),
        "peak_angle_deg": peak_angle,
        "final_angle_deg": final_angle,
        "peak_gain_deg": peak_gain,
        "final_gain_deg": final_gain,
        "maximum_normal_force_n": maximum_force,
        "maximum_force_step": force_peak_step,
        "force_over_0.5_n": maximum_force > 0.5,
        "contact_rate": contact_rate,
        "transition_contact_rate": transition_contact_rate,
        "hold_contact_rate": hold_contact_rate,
        "first_contact_loss_step": first_contact_loss,
        "minimum_ring_contact_distance_mm": min_dist_mm,
        "maximum_joint_tracking_errors_rad": max_errors,
        "final_joint_tracking_errors_rad": final_errors,
        "maximum_actuator_forces": max_actuator_forces,
        "actuator_force_limits": force_limits,
        "maximum_cube_axis_drift_mm": max_axis_drift,
        "warnings": warnings,
    }
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during breakthrough diagnostic")
    lines = [
        "test=Prompt 29 specified ring breakthrough diagnostic",
        f"trace={OUTPUT_CSV}",
        f"summary={json.dumps(summary, sort_keys=True)}",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
