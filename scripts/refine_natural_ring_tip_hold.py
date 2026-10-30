"""Locally refine the best static true-ring-tip topology using real hold dynamics."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from search_articulated_face_followthrough import MAIN_SCENE, SCENE
from search_global_natural_ring_tip_grasp import OUTPUT_LOG as PLANNER_LOG
from search_global_redundant_grasp import EXPECTED_MAIN_SHA256, EXPECTED_SCENE_SHA256
from search_natural_redundant_grasp import set_candidate
from validate_global_natural_ring_tip_grasp import OUTPUT_LOG as VALIDATION_LOG
from validate_poses import (
    PENETRATION_LIMIT,
    ROOT,
    finger_prefix,
    geom_name,
    hand_geoms,
    object_id,
    sha256,
)


OUTPUT_LOG = ROOT / "logs" / "prompt41_natural_ring_tip_hold_refinement.txt"
OUTPUT_POSE = ROOT / "poses" / "natural_ring_tip_hold_R01.json"
SEED = 41_201
GENERATIONS = 4
POPULATION = 160
ELITES = 16
HOLD_STEPS = 250
LAST_STEPS = 100
JOINT_NAMES = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)


def source_candidate() -> dict[str, object]:
    results = [
        json.loads(line.removeprefix("result="))
        for line in VALIDATION_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("result=")
    ]
    return max(results, key=lambda result: float(result["support_full_rate"]))["candidate"]


def evaluate(
    model: mujoco.MjModel,
    candidate: dict[str, object],
    vector: np.ndarray,
) -> dict[str, object]:
    data = mujoco.MjData(model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_qpos = int(model.jnt_qposadr[face_joint])
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    tip_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_tip")
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    hand_set = set(hand_geoms(model))
    joints = dict(candidate["joints"])
    for index, name in enumerate(JOINT_NAMES):
        joints[name] = float(vector[index])
    mount = dict(candidate["mount"])
    mount["mount_pitch"] = float(vector[16])
    mount["mount_roll"] = float(vector[17])
    cube_xyz = [float(value) for value in vector[18:21]]
    set_candidate(
        model,
        data,
        cube_qpos,
        face_qpos,
        cube_xyz,
        candidate["cube_quat_wxyz"],
        mount,
        joints,
    )
    start_cube = data.xpos[cube_body].copy()
    start_face = float(data.qpos[face_qpos])
    tip_history: list[bool] = []
    support_history: list[bool] = []
    maximum_drift = 0.0
    minimum_distance = float("inf")
    peak_tip_force = 0.0
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for _ in range(HOLD_STEPS):
            mujoco.mj_step(model, data)
            tip_contact = False
            prefixes: set[str] = set()
            tip_force = 0.0
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
                            prefixes.add(prefix)
                if face_geom in pair:
                    other = second if first == face_geom else first
                    if other == tip_geom:
                        tip_contact = True
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, index, force)
                        tip_force += max(0.0, float(force[0]))
            tip_history.append(tip_contact)
            support_history.append({"th", "if", "mf"}.issubset(prefixes))
            peak_tip_force = max(peak_tip_force, tip_force)
            maximum_drift = max(
                maximum_drift, float(np.max(np.abs(data.xpos[cube_body] - start_cube)))
            )
    finally:
        mujoco.set_mju_user_warning(old_handler)
    tip_rate = sum(tip_history[-LAST_STEPS:]) / LAST_STEPS
    support_rate = sum(support_history[-LAST_STEPS:]) / LAST_STEPS
    face_change = math.degrees(float(data.qpos[face_qpos]) - start_face)
    penetration = minimum_distance < -PENETRATION_LIMIT
    score = (
        8.0 * min(tip_rate, support_rate)
        + tip_rate
        + support_rate
        - 200.0 * maximum_drift
        - 0.2 * abs(face_change)
        - (10.0 if penetration or warnings else 0.0)
    )
    passed = (
        tip_rate >= 0.95
        and support_rate >= 0.95
        and maximum_drift <= 0.0005
        and abs(face_change) <= 2.0
        and peak_tip_force <= 0.35
        and not penetration
        and not warnings
    )
    return {
        "score": score,
        "passed": passed,
        "tip_last100_rate": tip_rate,
        "support_last100_rate": support_rate,
        "maximum_axis_drift_mm": 1000 * maximum_drift,
        "face_angle_change_deg": face_change,
        "peak_tip_force_n": peak_tip_force,
        "minimum_distance_m": minimum_distance,
        "penetration_over_0.002_m": penetration,
        "warnings": warnings,
        "vector": vector.tolist(),
        "joints": joints,
        "mount": mount,
        "cube_xyz": cube_xyz,
    }


def main() -> None:
    if sha256(MAIN_SCENE) != EXPECTED_MAIN_SHA256 or sha256(SCENE) != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch")
    candidate = source_candidate()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    center = np.array(
        [candidate["joints"][name] for name in JOINT_NAMES]
        + [candidate["mount"]["mount_pitch"], candidate["mount"]["mount_roll"]]
        + list(candidate["cube_xyz"]),
        dtype=float,
    )
    sigma = np.array([0.08] * 16 + [0.025, 0.025] + [0.0015] * 3)
    lower = np.empty_like(center)
    upper = np.empty_like(center)
    for index, name in enumerate(JOINT_NAMES):
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower[index], upper[index] = model.jnt_range[joint]
    lower[8:12] = np.maximum(lower[8:12], [0.0, -0.5, -0.1, -0.1])
    upper[8:12] = np.minimum(upper[8:12], [1.8, 0.5, 1.5, 1.6])
    lower[16:18], upper[16:18] = -0.2, 0.2
    lower[18:21] = center[18:21] - 0.004
    upper[18:21] = center[18:21] + 0.004
    rng = np.random.default_rng(SEED)
    all_results: list[dict[str, object]] = []
    for generation in range(GENERATIONS):
        vectors = np.clip(rng.normal(center, sigma, size=(POPULATION, len(center))), lower, upper)
        vectors[0] = center
        results = [evaluate(model, candidate, vector) for vector in vectors]
        results.sort(key=lambda result: float(result["score"]), reverse=True)
        all_results.extend(results)
        elites = np.asarray([result["vector"] for result in results[:ELITES]], dtype=float)
        center = np.mean(elites, axis=0)
        sigma = np.maximum(np.std(elites, axis=0), sigma * 0.45)
        print(
            f"generation={generation} best_score={results[0]['score']:.6f} "
            f"tip={results[0]['tip_last100_rate']:.3f} "
            f"support={results[0]['support_last100_rate']:.3f} "
            f"drift_mm={results[0]['maximum_axis_drift_mm']:.6f}"
        )
    all_results.sort(key=lambda result: float(result["score"]), reverse=True)
    passing = [result for result in all_results if result["passed"]]
    best = passing[0] if passing else all_results[0]
    if passing:
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "test_only": True,
                    "label": "natural_ring_tip_hold_R01",
                    "source_planner": str(PLANNER_LOG),
                    "cube_xyz": best["cube_xyz"],
                    "cube_quat_wxyz": candidate["cube_quat_wxyz"],
                    "mount": best["mount"],
                    "joints": best["joints"],
                    "measured": {key: value for key, value in best.items() if key not in {"vector", "joints", "mount", "cube_xyz"}},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    lines = [
        "test=natural true-ring-tip dynamic hold refinement",
        f"seed={SEED}",
        f"generations={GENERATIONS}",
        f"population={POPULATION}",
        f"hold_steps={HOLD_STEPS}",
        f"evaluated={len(all_results)}",
        f"best={json.dumps(best, sort_keys=True)}",
        f"passing_count={len(passing)}",
        f"NATURAL_RING_TIP_HOLD={'PASS' if passing else 'FAIL'}",
        f"promoted_pose={OUTPUT_POSE if passing else 'NONE'}",
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[5:]))


if __name__ == "__main__":
    main()
