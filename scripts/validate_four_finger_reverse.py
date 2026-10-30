"""Validate a four-finger scripted reversal from the 15.766-degree state."""

from __future__ import annotations

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


KF03_POSE = ROOT / "poses" / "articulated_face_followthrough_KF03.json"
OUTPUT_POSE = ROOT / "poses" / "articulated_face_four_finger_reverse_R01.json"
OUTPUT_LOG = ROOT / "logs" / "prompt28_four_finger_reverse_validation.txt"
REVERSE_DELTAS = {
    "if_pip": 0.20,
    "mf_pip": 0.20,
    "rf_mcp": -0.40,
    "rf_pip": 0.40,
    "rf_dip": 0.80,
    "th_ipl": 0.20,
}
TRANSITION_STEPS = 300
HOLD_STEPS = 500


def replay_to_kf03(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    replay_to_kf02(model, data)
    candidate = json.loads(KF03_POSE.read_text(encoding="utf-8"))
    start = {
        name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS
    }
    for step in range(int(candidate["search_steps"]) + int(candidate["hold_steps"])):
        if step < int(candidate["search_steps"]):
            blend = 0.5 - 0.5 * math.cos(
                math.pi * (step + 1) / int(candidate["search_steps"])
            )
            for name in RING_JOINTS:
                target = float(candidate["target_joints"][name])
                data.ctrl[actuator_id(model, name)] = start[name] + blend * (
                    target - start[name]
                )
        mujoco.mj_step(model, data)


def run_once(model: mujoco.MjModel) -> tuple[dict[str, object], dict[str, float]]:
    data = mujoco.MjData(model)
    replay_to_kf03(model, data)
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    hand_set = set(hand_geoms(model))
    start_angle = float(data.qpos[face_qpos])
    start_relative = data.xpos[cube_body] - data.xpos[palm_body]
    starts = {
        name: float(data.qpos[qpos_address(model, name)]) for name in REVERSE_DELTAS
    }
    targets = {name: starts[name] + delta for name, delta in REVERSE_DELTAS.items()}
    for name, value in targets.items():
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = model.jnt_range[joint]
        if not lower <= value <= upper:
            raise RuntimeError(f"reverse target outside range: {name}={value}")
    support: list[bool] = []
    maximum_relative_drift = 0.0
    minimum_distance = float("inf")
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for step in range(TRANSITION_STEPS + HOLD_STEPS):
            if step < TRANSITION_STEPS:
                blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / TRANSITION_STEPS)
                for name, target in targets.items():
                    data.ctrl[actuator_id(model, name)] = starts[name] + blend * (
                        target - starts[name]
                    )
            mujoco.mj_step(model, data)
            core_prefixes: set[str] = set()
            for contact in data.contact[: data.ncon]:
                first, second = int(contact.geom1), int(contact.geom2)
                if first in hand_set or second in hand_set:
                    minimum_distance = min(minimum_distance, float(contact.dist))
                if core_geom in {first, second}:
                    other = second if first == core_geom else first
                    if other in hand_set:
                        prefix = finger_prefix(geom_name(model, other))
                        if prefix:
                            core_prefixes.add(prefix)
            support.append({"th", "if", "mf"}.issubset(core_prefixes))
            relative = data.xpos[cube_body] - data.xpos[palm_body]
            maximum_relative_drift = max(
                maximum_relative_drift,
                float(np.linalg.norm(relative - start_relative)),
            )
    finally:
        mujoco.set_mju_user_warning(old_handler)
    final_angle = float(data.qpos[face_qpos])
    penetration = minimum_distance < -PENETRATION_LIMIT
    passed = (
        math.degrees(start_angle - final_angle) >= 10.0
        and abs(math.degrees(final_angle)) <= 1.0
        and all(support[-100:])
        and maximum_relative_drift < 0.015
        and not penetration
        and not warnings
    )
    return (
        {
            "start_angle_deg": math.degrees(start_angle),
            "final_angle_deg": math.degrees(final_angle),
            "reverse_angle_deg": math.degrees(start_angle - final_angle),
            "support_rate": sum(support) / len(support),
            "support_sustained_last_100": all(support[-100:]),
            "maximum_palm_relative_drift_mm": 1000 * maximum_relative_drift,
            "minimum_distance_m": minimum_distance,
            "penetration_over_0.002_m": penetration,
            "warnings": warnings,
            "pass": passed,
        },
        targets,
    )


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before four-finger reverse validation")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    run_1, targets = run_once(model)
    run_2, targets_2 = run_once(model)
    repeatable = run_1["pass"] and run_2["pass"] and targets == targets_2
    if repeatable:
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "test_only": True,
                    "label": "articulated_face_four_finger_reverse_R01",
                    "source_pose": str(KF03_POSE),
                    "transition_steps": TRANSITION_STEPS,
                    "hold_steps": HOLD_STEPS,
                    "deltas": REVERSE_DELTAS,
                    "target_joints": targets,
                    "validation": {"run_1": run_1, "run_2": run_2},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during four-finger reverse validation")
    lines = [
        "test=Prompt 28 four-finger articulated-face reverse",
        f"deltas={REVERSE_DELTAS}",
        f"run_1={run_1}",
        f"run_2={run_2}",
        f"REPEATABLE_FOUR_FINGER_REVERSE={'PASS' if repeatable else 'FAIL'}",
        f"output_pose={OUTPUT_POSE if repeatable else 'none'}",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
