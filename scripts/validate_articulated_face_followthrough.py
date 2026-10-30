"""Replay the selected follow-through from a clean state twice."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import (
    CONTACT_FORCE_THRESHOLD_N,
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    OUTPUT_POSE,
    RING_JOINTS,
    SCENE,
    distal,
    replay_to_rf02,
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


OUTPUT_LOG = ROOT / "logs" / "prompt25_face_followthrough_validation_revision_3.txt"
HOLD_STEPS = 500
MIN_ANGLE_DEG = 5.0
MAX_ANGLE_DEG = 10.0
MAX_LAST_100_RANGE_DEG = 0.25


def run_once(model: mujoco.MjModel, candidate: dict[str, object]) -> dict[str, object]:
    data = mujoco.MjData(model)
    replay_to_rf02(model, data)
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    hand_set = set(hand_geoms(model))
    baseline_relative = data.xpos[cube_body] - data.xpos[palm_body]
    start_qpos = {
        name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS
    }
    targets = {name: float(value) for name, value in candidate["target_joints"].items()}
    search_steps = int(candidate["search_steps"])
    total_steps = search_steps + HOLD_STEPS
    forces: list[float] = []
    support: list[bool] = []
    ring_core: list[bool] = []
    angles: list[float] = []
    maximum_relative_drift = 0.0
    minimum_distance = float("inf")
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(total_steps):
            if step < search_steps:
                blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / search_steps)
                for name in RING_JOINTS:
                    data.ctrl[actuator_id(model, name)] = start_qpos[name] + blend * (
                        targets[name] - start_qpos[name]
                    )
            mujoco.mj_step(model, data)
            core_prefixes: set[str] = set()
            ring_core_now = False
            normal_force = 0.0
            rotation = data.xmat[face_body].reshape(3, 3)
            center = data.xpos[face_body]
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
                        ring_core_now |= prefix == "rf"
                elif face_geom in pair:
                    other = second if first == face_geom else first
                    if other in hand_set:
                        name = geom_name(model, other)
                        local = rotation.T @ (np.asarray(contact.pos) - center)
                        on_side = (
                            max(abs(local[0]), abs(local[1])) >= 0.023
                            and abs(local[2]) <= 0.0085
                        )
                        if finger_prefix(name) == "rf" and distal(name) and on_side:
                            force = np.zeros(6)
                            mujoco.mj_contactForce(model, data, contact_index, force)
                            normal_force += max(0.0, float(force[0]))
            forces.append(normal_force)
            support.append({"th", "if", "mf"}.issubset(core_prefixes))
            ring_core.append(ring_core_now)
            angles.append(float(data.qpos[face_qpos]))
            relative = data.xpos[cube_body] - data.xpos[palm_body]
            maximum_relative_drift = max(
                maximum_relative_drift,
                float(np.linalg.norm(relative - baseline_relative)),
            )
    finally:
        mujoco.set_mju_user_warning(old_handler)

    transition_forces = forces[:search_steps]
    last_100_forces = forces[-100:]
    final_angle_deg = math.degrees(angles[-1])
    last_100_range_deg = math.degrees(max(angles[-100:]) - min(angles[-100:]))
    transition_contact_rate = sum(
        force >= CONTACT_FORCE_THRESHOLD_N for force in transition_forces
    ) / search_steps
    last_100_contact_rate = sum(
        force >= CONTACT_FORCE_THRESHOLD_N for force in last_100_forces
    ) / 100
    penetration = minimum_distance < -PENETRATION_LIMIT
    passed = (
        transition_contact_rate > 0.95
        and float(np.mean(transition_forces)) > 0.005
        and all(support)
        and not any(ring_core)
        and last_100_contact_rate == 1.0
        and MIN_ANGLE_DEG <= final_angle_deg <= MAX_ANGLE_DEG
        and last_100_range_deg <= MAX_LAST_100_RANGE_DEG
        and maximum_relative_drift < 0.015
        and not penetration
        and not warnings
    )
    return {
        "final_angle_deg": final_angle_deg,
        "maximum_angle_deg": math.degrees(max(angles)),
        "last_100_range_deg": last_100_range_deg,
        "transition_contact_rate": transition_contact_rate,
        "transition_average_force_n": float(np.mean(transition_forces)),
        "last_100_contact_rate": last_100_contact_rate,
        "last_100_average_force_n": float(np.mean(last_100_forces)),
        "support_all_steps": all(support),
        "ring_core_any": any(ring_core),
        "maximum_relative_drift_mm": 1000 * maximum_relative_drift,
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": penetration,
        "warnings": warnings,
        "pass": passed,
    }


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before follow-through validation")
    candidate = json.loads(OUTPUT_POSE.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    runs = [run_once(model, candidate), run_once(model, candidate)]
    repeatable = all(bool(run["pass"]) for run in runs)
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during follow-through validation")
    lines = [
        "test=Prompt 25 clean-state articulated-face follow-through validation",
        f"candidate={OUTPUT_POSE}",
        f"angle_gate_deg=[{MIN_ANGLE_DEG:.1f}, {MAX_ANGLE_DEG:.1f}]",
        f"last_100_stability_range_limit_deg={MAX_LAST_100_RANGE_DEG:.3f}",
    ]
    for index, run in enumerate(runs, start=1):
        lines.append(f"run_{index}={run}")
    lines.extend(
        [
            f"REPEATABLE_FOLLOWTHROUGH_GATE={'PASS' if repeatable else 'FAIL'}",
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
