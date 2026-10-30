"""Run the user-specified fine asymmetric KF03 grid with an edge-force cap."""

from __future__ import annotations

import itertools
import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import (
    CONTACT_FORCE_THRESHOLD_N,
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    RING_JOINTS,
    SCENE,
)
from search_articulated_face_kf03 import KF02_POSE, replay_to_kf02
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


OUTPUT_POSE = ROOT / "poses" / "articulated_face_followthrough_KF03_R1.json"
OUTPUT_LOG = ROOT / "logs" / "prompt27_face_kf03_R1_search.txt"
DELTAS_MCP = (0.0, 0.02, 0.04, 0.06, 0.08)
DELTAS_FINGER = (-0.06, -0.04, -0.02, 0.0, 0.02)
SEARCH_STEPS = 150
HOLD_STEPS = 500
MIN_ANGLE_DEG = 15.0
MAX_ANGLE_DEG = 20.0
MAX_AXIS_DRIFT_M = 0.0003
MAX_NORMAL_FORCE_N = 0.35
MIN_CONTACT_RATE = 0.95
MIN_AVERAGE_FORCE_N = 0.005
MAX_LAST_100_RANGE_DEG = 0.25


def evaluate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: dict[str, float],
    start_qpos: dict[str, float],
    baseline_angle: float,
    baseline_cube_xyz: np.ndarray,
) -> dict[str, object]:
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    hand_set = set(hand_geoms(model))
    forces: list[float] = []
    angles: list[float] = []
    support_steps = 0
    ring_core_steps = 0
    force_spike_steps = 0
    maximum_force = 0.0
    transition_axis_drift = np.zeros(3)
    full_axis_drift = np.zeros(3)
    minimum_distance = float("inf")
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(SEARCH_STEPS + HOLD_STEPS):
            if step < SEARCH_STEPS:
                blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / SEARCH_STEPS)
                for name in RING_JOINTS:
                    data.ctrl[actuator_id(model, name)] = start_qpos[name] + blend * (
                        targets[name] - start_qpos[name]
                    )
            mujoco.mj_step(model, data)
            rotation = data.xmat[face_body].reshape(3, 3)
            center = data.xpos[face_body]
            core_prefixes: set[str] = set()
            ring_core = False
            normal_force = 0.0
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
                        on_side = (
                            max(abs(local[0]), abs(local[1])) >= 0.023
                            and abs(local[2]) <= 0.0085
                        )
                        if on_side:
                            force = np.zeros(6)
                            mujoco.mj_contactForce(model, data, contact_index, force)
                            normal_force += max(0.0, float(force[0]))
            forces.append(normal_force)
            maximum_force = max(maximum_force, normal_force)
            force_spike_steps += normal_force > MAX_NORMAL_FORCE_N
            angles.append(float(data.qpos[face_qpos]))
            support_steps += {"th", "if", "mf"}.issubset(core_prefixes)
            ring_core_steps += ring_core
            drift = np.abs(data.xpos[cube_body] - baseline_cube_xyz)
            full_axis_drift = np.maximum(full_axis_drift, drift)
            if step < SEARCH_STEPS:
                transition_axis_drift = np.maximum(transition_axis_drift, drift)
    finally:
        mujoco.set_mju_user_warning(old_handler)

    transition_forces = forces[:SEARCH_STEPS]
    last_100_forces = forces[-100:]
    transition_contact_rate = sum(
        force >= CONTACT_FORCE_THRESHOLD_N for force in transition_forces
    ) / SEARCH_STEPS
    last_100_contact_rate = sum(
        force >= CONTACT_FORCE_THRESHOLD_N for force in last_100_forces
    ) / 100
    final_angle_deg = math.degrees(angles[-1])
    last_100_range_deg = math.degrees(max(angles[-100:]) - min(angles[-100:]))
    penetration = minimum_distance < -PENETRATION_LIMIT
    passed = (
        MIN_ANGLE_DEG <= final_angle_deg <= MAX_ANGLE_DEG
        and transition_contact_rate >= MIN_CONTACT_RATE
        and float(np.mean(transition_forces)) >= MIN_AVERAGE_FORCE_N
        and last_100_contact_rate >= MIN_CONTACT_RATE
        and support_steps == SEARCH_STEPS + HOLD_STEPS
        and ring_core_steps == 0
        and force_spike_steps == 0
        and last_100_range_deg <= MAX_LAST_100_RANGE_DEG
        and float(np.max(full_axis_drift)) <= MAX_AXIS_DRIFT_M
        and not penetration
        and not warnings
        and angles[-1] > baseline_angle
    )
    return {
        "final_angle_deg": final_angle_deg,
        "angle_gain_deg": math.degrees(angles[-1] - baseline_angle),
        "last_100_range_deg": last_100_range_deg,
        "transition_contact_rate": transition_contact_rate,
        "transition_average_force_n": float(np.mean(transition_forces)),
        "last_100_contact_rate": last_100_contact_rate,
        "last_100_average_force_n": float(np.mean(last_100_forces)),
        "maximum_normal_force_n": maximum_force,
        "force_spike_steps": force_spike_steps,
        "support_rate": support_steps / (SEARCH_STEPS + HOLD_STEPS),
        "ring_core_rate": ring_core_steps / (SEARCH_STEPS + HOLD_STEPS),
        "transition_max_axis_drift_mm": [1000 * float(value) for value in transition_axis_drift],
        "full_max_axis_drift_mm": [1000 * float(value) for value in full_axis_drift],
        "maximum_transition_axis_drift_mm": 1000 * float(np.max(transition_axis_drift)),
        "maximum_full_axis_drift_mm": 1000 * float(np.max(full_axis_drift)),
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": penetration,
        "warnings": warnings,
        "pass": passed,
    }


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before KF03_R1 search")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    kf02 = json.loads(KF02_POSE.read_text(encoding="utf-8"))
    fixed_rf_rot = float(kf02["target_joints"]["rf_rot"])
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    baseline_angle = float(data.qpos[face_qpos])
    baseline_cube_xyz = data.xpos[cube_body].copy()
    baseline_qpos = {
        name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS
    }
    residuals: dict[str, dict[str, float]] = {}
    for name, value in baseline_qpos.items():
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = map(float, model.jnt_range[joint])
        residuals[name] = {"actual": value, "to_lower": value - lower, "to_upper": upper - value}
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)
    results: list[dict[str, object]] = []
    for d_mcp, d_pip, d_dip in itertools.product(
        DELTAS_MCP, DELTAS_FINGER, DELTAS_FINGER
    ):
        deltas = {"rf_mcp": d_mcp, "rf_pip": d_pip, "rf_dip": d_dip}
        targets = {
            "rf_mcp": baseline_qpos["rf_mcp"] + d_mcp,
            "rf_rot": fixed_rf_rot,
            "rf_pip": baseline_qpos["rf_pip"] + d_pip,
            "rf_dip": baseline_qpos["rf_dip"] + d_dip,
        }
        if any(
            not model.jnt_range[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)][0]
            <= value
            <= model.jnt_range[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)][1]
            for name, value in targets.items()
        ):
            continue
        mujoco.mj_setState(model, data, state, state_spec)
        mujoco.mj_forward(model, data)
        result = evaluate(
            model,
            data,
            targets,
            baseline_qpos,
            baseline_angle,
            baseline_cube_xyz,
        )
        results.append({"deltas": deltas, "targets": targets, **result})
    passed = [result for result in results if result["pass"]]
    passed.sort(key=lambda result: float(result["final_angle_deg"]), reverse=True)
    ranked = sorted(results, key=lambda result: float(result["final_angle_deg"]), reverse=True)
    best = passed[0] if passed else None
    if best is not None:
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "test_only": True,
                    "label": "articulated_face_followthrough_KF03_R1",
                    "source_pose": str(KF02_POSE),
                    "search_steps": SEARCH_STEPS,
                    "hold_steps": HOLD_STEPS,
                    "fixed_rf_rot": fixed_rf_rot,
                    "baseline_face_angle_rad": baseline_angle,
                    "deltas": best["deltas"],
                    "target_joints": best["targets"],
                    "search_result": best,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during KF03_R1 search")
    half_size = [float(value) for value in model.geom_size[rf_geom]]
    lines = [
        "test=Prompt 27 user-specified fine asymmetric KF03_R1 search",
        f"candidates_evaluated={len(results)}",
        f"deltas_mcp={list(DELTAS_MCP)}",
        f"deltas_finger={list(DELTAS_FINGER)}",
        f"search_steps={SEARCH_STEPS}",
        f"hold_steps={HOLD_STEPS}",
        f"baseline_face_angle_deg={math.degrees(baseline_angle):+.9f}",
        f"baseline_ring_qpos={baseline_qpos}",
        f"joint_residuals_rad={residuals}",
        f"fixed_rf_rot_target_rad={fixed_rf_rot:+.9f}",
        f"rf_ds_collision_1_box_half_size_m={half_size}",
        f"rf_ds_collision_1_box_full_size_m={[2 * value for value in half_size]}",
        f"maximum_normal_force_n={MAX_NORMAL_FORCE_N:.6f}",
        f"maximum_axis_drift_mm={1000 * MAX_AXIS_DRIFT_M:.6f}",
        f"passing_candidates={len(passed)}",
        f"best_candidate={best or 'none'}",
        "top_10_by_final_angle:",
    ]
    for rank, result in enumerate(ranked[:10], start=1):
        lines.append(
            f"{rank:02d} deltas={result['deltas']} "
            f"final_deg={float(result['final_angle_deg']):+.6f} "
            f"contact={float(result['last_100_contact_rate']):.3f} "
            f"max_force_n={float(result['maximum_normal_force_n']):.6f} "
            f"spike_steps={int(result['force_spike_steps'])} "
            f"transition_axis_mm={result['transition_max_axis_drift_mm']} "
            f"full_axis_mm={result['full_max_axis_drift_mm']} "
            f"support={float(result['support_rate']):.3f} "
            f"pass={'YES' if result['pass'] else 'NO'}"
        )
    lines.extend(
        [
            f"KF03_R1_SEARCH={'PASS' if best is not None else 'FAIL'}",
            f"output_pose={OUTPUT_POSE if best is not None else 'none'}",
            f"main_scene_sha256_before={main_before}",
            f"main_scene_sha256_after={main_after}",
            f"test_scene_sha256_before={scene_before}",
            f"test_scene_sha256_after={scene_after}",
        ]
    )
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
