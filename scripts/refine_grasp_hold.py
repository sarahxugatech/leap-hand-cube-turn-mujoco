"""Locally refine a static LEAP grasp for short passive-hold robustness."""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from validate_poses import (
    EXPECTED_SCENE_SHA256,
    HAND_JOINTS,
    MAIN_SCENE,
    ROOT,
    actuator_id,
    finger_prefix,
    geom_name,
    hand_geoms,
    object_id,
    qpos_address,
    sha256,
    test_scene_copy,
)


SEED = 12031
POPULATION = 256
GENERATIONS = 10
ELITE_FRACTION = 0.10
SCREEN_STEPS = 100
PENETRATION_LIMIT = 0.002
SOURCE_POSE = ROOT / "poses" / "grasp_optimizer_C02.json"
OUTPUT_LOG = ROOT / "logs" / "prompt12_hold_refinement.txt"


def main() -> None:
    started = time.monotonic()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    source_pose = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    source_xml = MAIN_SCENE.read_text(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    history: list[dict[str, object]] = []
    generation_lines: list[str] = []

    with TemporaryDirectory(prefix="prompt12_refine_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(source_xml, Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        hand = hand_geoms(model)
        hand_set = set(hand)
        joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HAND_JOINTS]
        joint_low = np.array([model.jnt_range[joint, 0] for joint in joint_ids])
        joint_high = np.array([model.jnt_range[joint, 1] for joint in joint_ids])
        mean = np.array(
            [source_pose["joints"][name] for name in HAND_JOINTS] + source_pose["cube_start_xyz"],
            dtype=float,
        )
        std = np.concatenate((np.full(16, 0.10), np.full(3, 0.004)))
        lower = np.concatenate((joint_low, mean[16:19] - 0.012))
        upper = np.concatenate((joint_high, mean[16:19] + 0.012))

        def contacts_and_penetration() -> tuple[bool, set[str], float]:
            thumb = False
            fingers: set[str] = set()
            deepest = float("inf")
            for contact in data.contact[: data.ncon]:
                first = int(contact.geom1)
                second = int(contact.geom2)
                pair = {first, second}
                if cube_geom in pair:
                    other = second if first == cube_geom else first
                    if other in hand_set:
                        prefix = finger_prefix(geom_name(model, other))
                        thumb = thumb or prefix == "th"
                        if prefix in {"if", "mf", "rf"}:
                            fingers.add(prefix)
                        deepest = min(deepest, float(contact.dist))
                elif table_geom in pair:
                    other = second if first == table_geom else first
                    if other in hand_set:
                        deepest = min(deepest, float(contact.dist))
                elif first in hand_set and second in hand_set:
                    first_prefix = finger_prefix(geom_name(model, first))
                    second_prefix = finger_prefix(geom_name(model, second))
                    if (
                        first_prefix in {"if", "mf", "rf", "th"}
                        and second_prefix in {"if", "mf", "rf", "th"}
                        and first_prefix != second_prefix
                    ):
                        deepest = min(deepest, float(contact.dist))
            return thumb, fingers, deepest

        def evaluate(vector: np.ndarray) -> dict[str, object]:
            mujoco.mj_resetData(model, data)
            for name, value in source_pose["mount"].items():
                data.qpos[qpos_address(model, name)] = value
                data.ctrl[actuator_id(model, name)] = value
            for name, value in zip(HAND_JOINTS, vector[:16], strict=True):
                data.qpos[qpos_address(model, name)] = value
                data.ctrl[actuator_id(model, name)] = value
            data.qpos[cube_qpos : cube_qpos + 7] = (*vector[16:19], 1, 0, 0, 0)
            mujoco.mj_forward(model, data)
            cube_start = data.xpos[cube_body].copy()
            thumb_history: list[bool] = []
            final_fingers: set[str] = set()
            deepest = float("inf")
            maximum_displacement = 0.0
            initial_thumb = False
            initial_fingers: set[str] = set()
            for step in range(SCREEN_STEPS + 1):
                thumb, fingers, step_deepest = contacts_and_penetration()
                if step == 0:
                    initial_thumb, initial_fingers = thumb, fingers.copy()
                thumb_history.append(thumb)
                final_fingers = fingers
                deepest = min(deepest, step_deepest)
                maximum_displacement = max(
                    maximum_displacement,
                    float(np.linalg.norm(data.xpos[cube_body] - cube_start)),
                )
                if step < SCREEN_STEPS:
                    mujoco.mj_step(model, data)
            thumb_fraction = float(np.mean(thumb_history[-20:]))
            displacement_mm = maximum_displacement * 1000.0
            score = 40.0 * thumb_fraction + 12.0 * min(len(final_fingers), 2)
            score += 4.0 * initial_thumb + 2.0 * min(len(initial_fingers), 2)
            score -= 0.8 * displacement_mm
            if deepest < -PENETRATION_LIMIT:
                score -= 10000.0 + 1000.0 * (-deepest - PENETRATION_LIMIT) * 1000.0
            passed = (
                thumb_fraction == 1.0
                and len(final_fingers) >= 2
                and maximum_displacement < 0.015
                and deepest >= -PENETRATION_LIMIT
            )
            return {
                "vector": vector.copy(),
                "score": score,
                "thumb_fraction": thumb_fraction,
                "final_fingers": sorted(final_fingers),
                "max_displacement_m": maximum_displacement,
                "deepest_m": deepest,
                "screen_pass": passed,
            }

        for generation in range(GENERATIONS):
            population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), lower, upper)
            population[0] = np.clip(mean, lower, upper)
            results = [evaluate(vector) for vector in population]
            results.sort(key=lambda row: float(row["score"]), reverse=True)
            history.extend(results[:20])
            elite_count = max(2, int(POPULATION * ELITE_FRACTION))
            elites = np.stack([row["vector"] for row in results[:elite_count]])
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), np.concatenate((np.full(16, 0.01), np.full(3, 0.0005))))
            best = results[0]
            generation_lines.append(
                f"generation={generation + 1} score={best['score']:.6f} "
                f"thumb_last20={best['thumb_fraction']:.2f} fingers={best['final_fingers']} "
                f"max_disp_mm={best['max_displacement_m'] * 1000.0:.3f} "
                f"deepest_m={best['deepest_m']:+.9f} pass={best['screen_pass']}"
            )

    history.sort(key=lambda row: float(row["score"]), reverse=True)
    passing: list[dict[str, object]] = []
    seen: set[tuple[float, ...]] = set()
    for row in history:
        key = tuple(np.round(row["vector"], 6))
        if key not in seen and row["screen_pass"]:
            seen.add(key)
            passing.append(row)
        if len(passing) == 5:
            break
    pose_paths: list[Path] = []
    for index, row in enumerate(passing, start=1):
        vector = row["vector"]
        path = ROOT / "poses" / f"grasp_hold_refined_R{index:02d}.json"
        payload = {
            "test_only": True,
            "label": path.stem,
            "source_pose": str(SOURCE_POSE),
            "cube_start_xyz": vector[16:19].tolist(),
            "cube_start_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "mount": source_pose["mount"],
            "joints": dict(zip(HAND_JOINTS, vector[:16].tolist(), strict=True)),
            "screen": {
                "steps": SCREEN_STEPS,
                "thumb_last20_fraction": row["thumb_fraction"],
                "final_fingers": row["final_fingers"],
                "max_displacement_m": row["max_displacement_m"],
                "deepest_distance_m": row["deepest_m"],
                "pass": True,
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        pose_paths.append(path)

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during hold refinement")
    lines = [
        "test=Prompt 12 local dynamic hold refinement",
        f"source_pose={SOURCE_POSE}",
        "method=CEM-style local sampling; not MPC; not learned",
        f"seed={SEED}",
        f"population={POPULATION}",
        f"generations={GENERATIONS}",
        f"elite_fraction={ELITE_FRACTION}",
        f"screen_steps={SCREEN_STEPS}",
        f"elapsed_s={time.monotonic() - started:.3f}",
        f"screen_pass_candidates={len(passing)}",
        f"pose_files={[str(path) for path in pose_paths] or 'none'}",
        "cube_hand_attachment=NO",
        "cube_qpos_updates_during_run=0",
        "lift_run=NO",
        "rotation_run=NO",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
        "",
        *generation_lines,
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
