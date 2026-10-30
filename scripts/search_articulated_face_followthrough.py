"""Search one safe ring-finger follow-through keyframe after RF02."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from validate_poses import (
    PENETRATION_LIMIT,
    ROOT,
    actuator_id,
    finger_prefix,
    geom_name,
    hand_geoms,
    load_pose,
    object_id,
    qpos_address,
    sha256,
)


MAIN_SCENE = ROOT / "scenes" / "table_cube.xml"
SCENE = ROOT / "scenes" / "table_articulated_face_test.xml"
SOURCE_POSE = ROOT / "poses" / "articulated_grasp_R01.json"
RF02_POSE = ROOT / "poses" / "articulated_face_reach_RF02.json"
OUTPUT_LOG = ROOT / "logs" / "prompt25_face_followthrough_search_revision_3.txt"
OUTPUT_POSE = ROOT / "poses" / "articulated_face_followthrough_KF02_R4.json"
EXPECTED_MAIN_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
EXPECTED_SCENE_SHA256 = "baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb"
RING_JOINTS = ("rf_mcp", "rf_rot", "rf_pip", "rf_dip")
SEARCH_JOINTS = ("rf_mcp", "rf_pip", "rf_dip")
DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)
SEARCH_STEPS = 150
EVALUATION_HOLD_STEPS = 500
CONTACT_FORCE_THRESHOLD_N = 0.001
MIN_CONTACT_RATE = 0.95
MIN_AVERAGE_FORCE_N = 0.005
MIN_FINAL_ANGLE_DEG = 5.0
MAX_FINAL_ANGLE_DEG = 10.0
MAX_LAST_100_RANGE_DEG = 0.25


def distal(name: str) -> bool:
    return "_md_" in name or "_ds_" in name or name.endswith("_tip")


def replay_to_rf02(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    payload = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    candidate = json.loads(RF02_POSE.read_text(encoding="utf-8"))
    mount, joints, _, _ = load_pose(SOURCE_POSE, model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_qpos = int(model.jnt_qposadr[face_joint])
    data.qpos[cube_qpos : cube_qpos + 7] = (
        *payload["cube_start_xyz"],
        *payload["cube_start_quat_wxyz"],
    )
    data.qpos[face_qpos] = float(payload["face_turn_rad"])
    for name, value in mount.items() | joints.items():
        data.qpos[qpos_address(model, name)] = value
        data.ctrl[actuator_id(model, name)] = value
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    for step in range(800):
        phase = min((step + 1) / 300, 1.0)
        blend = 0.5 - 0.5 * math.cos(math.pi * phase)
        data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
        mujoco.mj_step(model, data)
    target = candidate["target_joints"]
    for step in range(int(candidate["transition_steps"]) + 500):
        if step < int(candidate["transition_steps"]):
            phase = (step + 1) / int(candidate["transition_steps"])
            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
            for name in RING_JOINTS:
                data.ctrl[actuator_id(model, name)] = joints[name] + blend * (
                    float(target[name]) - joints[name]
                )
        mujoco.mj_step(model, data)


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before follow-through search")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_to_rf02(model, data)

    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    hand_set = set(hand_geoms(model))
    baseline_angle = float(data.qpos[face_qpos])
    baseline_relative = data.xpos[cube_body] - data.xpos[palm_body]
    baseline_qpos = {
        name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS
    }

    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)
    results: list[dict[str, object]] = []

    for delta_values in itertools.product(DELTAS, repeat=3):
        if delta_values == (0.0, 0.0, 0.0):
            continue
        deltas = dict(zip(SEARCH_JOINTS, delta_values, strict=True))
        targets = baseline_qpos | {
            name: baseline_qpos[name] + deltas[name] for name in SEARCH_JOINTS
        }
        in_range = True
        for name, value in targets.items():
            joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lower, upper = model.jnt_range[joint]
            in_range &= bool(lower <= value <= upper)
        if not in_range:
            continue

        mujoco.mj_setState(model, data, state, state_spec)
        mujoco.mj_forward(model, data)
        forces: list[float] = []
        angle_history: list[float] = []
        support_steps = 0
        ring_core_steps = 0
        max_relative_drift = 0.0
        minimum_distance = float("inf")
        warnings: list[str] = []
        old_handler = mujoco.set_mju_user_warning(warnings.append)
        try:
            for step in range(SEARCH_STEPS + EVALUATION_HOLD_STEPS):
                if step < SEARCH_STEPS:
                    blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / SEARCH_STEPS)
                    for name in RING_JOINTS:
                        data.ctrl[actuator_id(model, name)] = baseline_qpos[name] + blend * (
                            targets[name] - baseline_qpos[name]
                        )
                mujoco.mj_step(model, data)

                core_prefixes: set[str] = set()
                ring_core = False
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
                            ring_core |= prefix == "rf"
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
                    elif table_geom in pair:
                        pass
                forces.append(normal_force)
                angle_history.append(float(data.qpos[face_qpos]))
                support_steps += {"th", "if", "mf"}.issubset(core_prefixes)
                ring_core_steps += ring_core
                relative = data.xpos[cube_body] - data.xpos[palm_body]
                max_relative_drift = max(
                    max_relative_drift,
                    float(np.linalg.norm(relative - baseline_relative)),
                )
        finally:
            mujoco.set_mju_user_warning(old_handler)

        transition_forces = forces[:SEARCH_STEPS]
        last_100_forces = forces[-100:]
        contact_rate = sum(
            force >= CONTACT_FORCE_THRESHOLD_N for force in transition_forces
        ) / SEARCH_STEPS
        last_100_contact_rate = sum(
            force >= CONTACT_FORCE_THRESHOLD_N for force in last_100_forces
        ) / 100
        average_force = float(np.mean(transition_forces))
        final_angle = float(data.qpos[face_qpos])
        last_100_range = max(angle_history[-100:]) - min(angle_history[-100:])
        penetration = minimum_distance < -PENETRATION_LIMIT
        admissible = (
            contact_rate > MIN_CONTACT_RATE
            and average_force > MIN_AVERAGE_FORCE_N
            and support_steps == SEARCH_STEPS + EVALUATION_HOLD_STEPS
            and ring_core_steps == 0
            and last_100_contact_rate == 1.0
            and MIN_FINAL_ANGLE_DEG <= math.degrees(final_angle) <= MAX_FINAL_ANGLE_DEG
            and math.degrees(last_100_range) <= MAX_LAST_100_RANGE_DEG
            and max_relative_drift < 0.015
            and not penetration
            and not warnings
            and final_angle > baseline_angle
        )
        results.append(
            {
                "deltas": deltas,
                "targets": targets,
                "final_angle_rad": final_angle,
                "angle_gain_rad": final_angle - baseline_angle,
                "average_force_n": average_force,
                "contact_rate": contact_rate,
                "last_100_contact_rate": last_100_contact_rate,
                "last_100_average_force_n": float(np.mean(last_100_forces)),
                "last_100_range_deg": math.degrees(last_100_range),
                "support_rate": support_steps / (SEARCH_STEPS + EVALUATION_HOLD_STEPS),
                "ring_core_rate": ring_core_steps / (SEARCH_STEPS + EVALUATION_HOLD_STEPS),
                "max_relative_drift_m": max_relative_drift,
                "minimum_distance_m": minimum_distance,
                "warnings": warnings,
                "admissible": admissible,
            }
        )

    admissible_results = [result for result in results if result["admissible"]]
    admissible_results.sort(key=lambda result: float(result["final_angle_rad"]), reverse=True)
    ranked = sorted(results, key=lambda result: float(result["final_angle_rad"]), reverse=True)
    selectable = admissible_results
    best = selectable[0] if selectable else None
    if best is not None:
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "test_only": True,
                    "label": "articulated_face_followthrough_KF02_R4",
                    "source_pose": str(RF02_POSE),
                    "search_steps": SEARCH_STEPS,
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
        raise RuntimeError("scene changed during follow-through search")
    geom_type = mujoco.mjtGeom(int(model.geom_type[rf_geom])).name
    lines = [
        "test=Prompt 25 deterministic articulated-face follow-through grid search",
        f"candidates_evaluated={len(results)}",
        f"search_steps={SEARCH_STEPS}",
        f"evaluation_hold_steps={EVALUATION_HOLD_STEPS}",
        f"deltas_rad={list(DELTAS)}",
        f"baseline_face_angle_deg={math.degrees(baseline_angle):+.9f}",
        f"baseline_ring_qpos={baseline_qpos}",
        f"rf_ds_collision_1_geom_type={geom_type}",
        f"contact_force_threshold_n={CONTACT_FORCE_THRESHOLD_N:.6f}",
        f"minimum_contact_rate={MIN_CONTACT_RATE:.6f}",
        f"minimum_average_force_n={MIN_AVERAGE_FORCE_N:.6f}",
        f"final_angle_gate_deg=[{MIN_FINAL_ANGLE_DEG:.6f}, {MAX_FINAL_ANGLE_DEG:.6f}]",
        f"admissible_candidates={len(admissible_results)}",
        f"selectable_candidates={len(selectable)}",
        f"best_candidate={best or 'none'}",
        "top_10_by_final_angle:",
    ]
    for rank, result in enumerate(ranked[:10], start=1):
        lines.append(
            f"{rank:02d} deltas={result['deltas']} "
            f"final_deg={math.degrees(float(result['final_angle_rad'])):+.6f} "
            f"gain_deg={math.degrees(float(result['angle_gain_rad'])):+.6f} "
            f"force_n={float(result['average_force_n']):.6f} "
            f"contact_rate={float(result['contact_rate']):.6f} "
            f"last100_contact_rate={float(result['last_100_contact_rate']):.6f} "
            f"support_rate={float(result['support_rate']):.6f} "
            f"ring_core_rate={float(result['ring_core_rate']):.6f} "
            f"drift_mm={1000 * float(result['max_relative_drift_m']):.6f} "
            f"min_dist_m={float(result['minimum_distance_m']):+.9f} "
            f"admissible={'YES' if result['admissible'] else 'NO'}"
        )
    lines.extend(
        [
            f"FOLLOWTHROUGH_SEARCH={'PASS' if best is not None else 'FAIL'}",
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
