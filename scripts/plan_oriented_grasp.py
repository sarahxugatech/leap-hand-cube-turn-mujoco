"""Search a TEST-only LEAP grasp with a redesigned palm orientation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from plan_grasp_candidates import cross_finger_pairs
from validate_poses import (
    EXPECTED_SCENE_SHA256,
    HAND_JOINTS,
    MAIN_SCENE,
    PENETRATION_LIMIT,
    ROOT,
    finger_prefix,
    geom_name,
    hand_geoms,
    object_id,
    qpos_address,
    sha256,
    test_scene_copy,
)


SEED = 16001
POPULATION = 1024
GENERATIONS = 14
ELITE_FRACTION = 0.08
TIME_BUDGET_S = 45.0
ORIENTATION_MIN_RAD = math.radians(10.0)
SIDE_MARGIN_M = 0.005
BASE_MOUNT = {"mount_x": -0.0425, "mount_y": -0.007, "mount_z": -0.219}
BASE_ANGLES = np.array([0.10, -0.10, 0.0])
OUTPUT_LOG = ROOT / "logs" / "prompt16_oriented_grasp_planner.txt"


def distal(name: str) -> bool:
    return "_md_" in name or "_ds_" in name or name.endswith("_tip")


def main() -> None:
    started = time.monotonic()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")

    source_pose = json.loads((ROOT / "poses" / "grasp_hold_refined_R01.json").read_text())
    dual_pose = json.loads((ROOT / "poses" / "dual_side_D01.json").read_text())
    mean_joints = np.array([dual_pose["target_joints"][name] for name in HAND_JOINTS])
    mean_cube = np.asarray(source_pose["cube_start_xyz"], dtype=float)
    rng = np.random.default_rng(SEED)
    history: list[dict[str, object]] = []
    generation_lines: list[str] = []

    with TemporaryDirectory(prefix="prompt16_oriented_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(MAIN_SCENE.read_text(), Path(temporary), enable_yaw=True)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        hand = hand_geoms(model)
        hand_set = set(hand)
        pairs = cross_finger_pairs(model, hand)
        groups = {
            prefix: [geom for geom in hand if finger_prefix(geom_name(model, geom)) == prefix]
            for prefix in ("if", "mf", "rf", "th")
        }
        joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HAND_JOINTS]
        joint_low = np.array([model.jnt_range[joint, 0] for joint in joint_ids])
        joint_high = np.array([model.jnt_range[joint, 1] for joint in joint_ids])

        mean = np.concatenate((mean_joints, BASE_ANGLES, mean_cube))
        std = np.concatenate((np.full(16, 0.32), [0.16, 0.16, 0.22], [0.014] * 3))
        lower = np.concatenate((joint_low, [-0.35, -0.35, -0.523599], mean_cube - [0.035, 0.035, 0.035]))
        upper = np.concatenate((joint_high, [0.35, 0.35, 0.523599], mean_cube + [0.035, 0.035, 0.035]))

        for name, value in BASE_MOUNT.items():
            data.qpos[qpos_address(model, name)] = value
        data.qpos[cube_qpos + 3 : cube_qpos + 7] = (1, 0, 0, 0)

        def evaluate(vector: np.ndarray) -> dict[str, object]:
            for name, value in zip(HAND_JOINTS, vector[:16], strict=True):
                data.qpos[qpos_address(model, name)] = value
            for name, value in zip(("mount_pitch", "mount_roll", "mount_yaw"), vector[16:19], strict=True):
                data.qpos[qpos_address(model, name)] = value
            data.qpos[cube_qpos : cube_qpos + 3] = vector[19:22]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            distances = {
                prefix: min(float(mujoco.mj_geomDistance(model, data, cube_geom, geom, 0.10, None)) for geom in group)
                for prefix, group in groups.items()
            }
            contacts: set[str] = set()
            side_contacts: set[str] = set()
            deepest_cube = float("inf")
            deepest_table = float("inf")
            deepest_self = float("inf")
            cube_z = float(data.xpos[cube_body, 2])
            for contact in data.contact[: data.ncon]:
                first, second = int(contact.geom1), int(contact.geom2)
                pair = {first, second}
                if cube_geom in pair:
                    other = second if first == cube_geom else first
                    if other in hand_set:
                        name = geom_name(model, other)
                        contacts.add(name)
                        deepest_cube = min(deepest_cube, float(contact.dist))
                        if (
                            finger_prefix(name) in {"if", "mf", "rf"}
                            and distal(name)
                            and float(contact.pos[2]) > cube_z - 0.025 + SIDE_MARGIN_M
                        ):
                            side_contacts.add(name)
                elif table_geom in pair:
                    other = second if first == table_geom else first
                    if other in hand_set:
                        deepest_table = min(deepest_table, float(contact.dist))
                elif first in hand_set and second in hand_set:
                    a, b = finger_prefix(geom_name(model, first)), finger_prefix(geom_name(model, second))
                    if a in {"if", "mf", "rf", "th"} and b in {"if", "mf", "rf", "th"} and a != b:
                        deepest_self = min(deepest_self, float(contact.dist))

            prefixes = {finger_prefix(name) for name in contacts}
            fingers = sorted(prefix for prefix in prefixes if prefix in {"if", "mf", "rf"})
            side_prefixes = {finger_prefix(name) for name in side_contacts}
            orientation_delta = float(np.linalg.norm(vector[16:19] - BASE_ANGLES))
            support_mm = np.array([distances["th"], *sorted(distances[p] for p in ("if", "mf", "rf"))[:2]]) * 1000
            score = -float(np.sum(np.abs(support_mm + 0.2)))
            score += 12.0 * ("th" in prefixes) + 6.0 * len(fingers) + 10.0 * len(side_prefixes)
            score += min(math.degrees(orientation_delta), 25.0) * 0.15
            safe = min(deepest_cube, deepest_table, deepest_self) >= -PENETRATION_LIMIT
            changed = orientation_delta >= ORIENTATION_MIN_RAD
            gate = "th" in prefixes and len(fingers) >= 2 and bool(side_prefixes) and safe and changed
            if not safe:
                score -= 10000.0
            if not changed:
                score -= 100.0
            return {
                "vector": vector.copy(), "score": score, "contacts": sorted(contacts),
                "side_contacts": sorted(side_contacts), "fingers": fingers, "thumb": "th" in prefixes,
                "orientation_delta": orientation_delta, "deepest_cube": deepest_cube,
                "deepest_table": deepest_table, "deepest_self": deepest_self, "gate": gate,
            }

        for generation in range(GENERATIONS):
            if time.monotonic() - started >= TIME_BUDGET_S:
                break
            population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), lower, upper)
            results = [evaluate(vector) for vector in population]
            results.sort(key=lambda row: float(row["score"]), reverse=True)
            history.extend(results[:40])
            elites = np.stack([row["vector"] for row in results[: max(2, int(POPULATION * ELITE_FRACTION))]])
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), np.concatenate((np.full(16, 0.025), [0.015] * 3, [0.0015] * 3)))
            best = results[0]
            generation_lines.append(
                f"generation={generation + 1} score={best['score']:.6f} gate={best['gate']} "
                f"angles_deg={np.rad2deg(best['vector'][16:19]).tolist()} fingers={best['fingers']} "
                f"side={best['side_contacts']} cube={best['vector'][19:22].tolist()}"
            )

    history.sort(key=lambda row: float(row["score"]), reverse=True)
    passing: list[dict[str, object]] = []
    seen: set[tuple[float, ...]] = set()
    for row in history:
        if not row["gate"]:
            continue
        key = tuple(np.round(row["vector"], 5))
        if key in seen:
            continue
        seen.add(key)
        passing.append(row)
        if len(passing) == 5:
            break

    pose_paths: list[Path] = []
    for index, row in enumerate(passing, start=1):
        vector = row["vector"]
        path = ROOT / "poses" / f"oriented_grasp_O{index:02d}.json"
        payload = {
            "test_only": True,
            "label": f"oriented_grasp_O{index:02d}",
            "cube_start_xyz": vector[19:22].tolist(),
            "cube_start_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "mount": {**BASE_MOUNT, "mount_pitch": float(vector[16]), "mount_roll": float(vector[17])},
            "mount_yaw_degrees": float(math.degrees(vector[18])),
            "joints": dict(zip(HAND_JOINTS, vector[:16].tolist(), strict=True)),
            "static_gate": {
                "thumb": row["thumb"], "non_thumb_fingers": row["fingers"],
                "contacts": row["contacts"], "distal_side_contacts": row["side_contacts"],
                "orientation_delta_degrees": math.degrees(row["orientation_delta"]),
                "minimum_hand_cube_m": row["deepest_cube"],
                "minimum_hand_table_m": row["deepest_table"],
                "minimum_finger_finger_m": row["deepest_self"], "pass": True,
            },
            "optimizer": {"kind": "static orientation/grasp CEM (not MPC, not learned)", "seed": SEED,
                          "population": POPULATION, "generations": GENERATIONS,
                          "elite_fraction": ELITE_FRACTION, "time_budget_s": TIME_BUDGET_S},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        pose_paths.append(path)

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during orientation planning")
    lines = [
        "test=Prompt 16 TEST-only mount-orientation grasp redesign",
        f"seed={SEED}", f"population={POPULATION}", f"generations_completed={len(generation_lines)}",
        f"elapsed_s={time.monotonic() - started:.3f}",
        "variables=16 LEAP joints + mount pitch/roll/yaw + initial cube XYZ",
        "dynamic_steps_run=0", "rotation_run=NO", f"static_candidates={len(passing)}",
        f"pose_files={[str(path) for path in pose_paths] or 'none'}",
        f"main_scene_sha256_before={before_hash}", f"main_scene_sha256_after={after_hash}", "",
        "generation_summary:", *generation_lines,
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("generation_summary:")]))


if __name__ == "__main__":
    main()
