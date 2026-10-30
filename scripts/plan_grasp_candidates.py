"""Sample LEAP right-hand held-grasp candidates in a frozen TEST scene."""

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
    finger_prefix,
    geom_name,
    hand_geoms,
    object_id,
    qpos_address,
    sha256,
    test_scene_copy,
)


SEED = 12030
POPULATION = 768
GENERATIONS = 12
ELITE_FRACTION = 0.10
TIME_BUDGET_S = 45.0
PENETRATION_LIMIT = 0.002
OUTPUT_LOG = ROOT / "logs" / "prompt12_grasp_optimizer.txt"
MOUNT = {
    "mount_x": -0.0425,
    "mount_y": -0.0070,
    "mount_z": -0.2190,
    "mount_pitch": 0.10,
    "mount_roll": -0.10,
}


def cross_finger_pairs(model: mujoco.MjModel, geoms: list[int]) -> list[tuple[int, int]]:
    return [
        (first, second)
        for index, first in enumerate(geoms)
        for second in geoms[index + 1 :]
        if finger_prefix(geom_name(model, first)) in {"if", "mf", "rf", "th"}
        and finger_prefix(geom_name(model, second)) in {"if", "mf", "rf", "th"}
        and finger_prefix(geom_name(model, first)) != finger_prefix(geom_name(model, second))
    ]


def main() -> None:
    started = time.monotonic()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    baseline = json.loads((ROOT / "poses" / "power_wrap_B_thumb_T3.json").read_text(encoding="utf-8"))
    mean_joints = np.array([baseline["joints"][name] for name in HAND_JOINTS], dtype=float)

    source = MAIN_SCENE.read_text(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    history: list[dict[str, object]] = []
    generation_lines = []
    with TemporaryDirectory(prefix="prompt12_grasp_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(source, Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        geoms = hand_geoms(model)
        geom_set = set(geoms)
        groups = {
            prefix: [geom for geom in geoms if finger_prefix(geom_name(model, geom)) == prefix]
            for prefix in ("if", "mf", "rf", "th")
        }
        pairs = cross_finger_pairs(model, geoms)
        joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HAND_JOINTS]
        joint_low = np.array([model.jnt_range[joint, 0] for joint in joint_ids])
        joint_high = np.array([model.jnt_range[joint, 1] for joint in joint_ids])
        mean = np.concatenate((mean_joints, np.array([-0.035, 0.0, 0.150])))
        std = np.concatenate((np.full(16, 0.40), np.array([0.025, 0.025, 0.020])))
        lower = np.concatenate((joint_low, np.array([-0.10, -0.08, 0.105])))
        upper = np.concatenate((joint_high, np.array([0.05, 0.08, 0.200])))

        for name, value in MOUNT.items():
            data.qpos[qpos_address(model, name)] = value
        data.qpos[cube_qpos + 3 : cube_qpos + 7] = (1, 0, 0, 0)

        def evaluate(vector: np.ndarray) -> dict[str, object]:
            for name, value in zip(HAND_JOINTS, vector[:16], strict=True):
                data.qpos[qpos_address(model, name)] = value
            data.qpos[cube_qpos : cube_qpos + 3] = vector[16:19]
            mujoco.mj_forward(model, data)
            distances = {
                prefix: min(
                    float(mujoco.mj_geomDistance(model, data, cube_geom, geom, 0.10, None))
                    for geom in group
                )
                for prefix, group in groups.items()
            }
            table_distance = min(
                float(mujoco.mj_geomDistance(model, data, table_geom, geom, 0.10, None))
                for geom in geoms
            )
            contact_hand_cube = float("inf")
            contact_cross_finger = float("inf")
            for contact in data.contact[: data.ncon]:
                first = int(contact.geom1)
                second = int(contact.geom2)
                contact_geoms = {first, second}
                if cube_geom in contact_geoms:
                    other = second if first == cube_geom else first
                    if other in geom_set:
                        contact_hand_cube = min(contact_hand_cube, float(contact.dist))
                elif first in geom_set and second in geom_set:
                    first_prefix = finger_prefix(geom_name(model, first))
                    second_prefix = finger_prefix(geom_name(model, second))
                    if (
                        first_prefix in {"if", "mf", "rf", "th"}
                        and second_prefix in {"if", "mf", "rf", "th"}
                        and first_prefix != second_prefix
                    ):
                        contact_cross_finger = min(contact_cross_finger, float(contact.dist))
            ordered_fingers = sorted(distances[prefix] for prefix in ("if", "mf", "rf"))[:2]
            support_mm = np.array([distances["th"], *ordered_fingers]) * 1000.0
            score = -float(np.sum(np.abs(support_mm + 0.2)))
            score += 8.0 * float(distances["th"] <= 0.0005)
            score += 4.0 * sum(distances[prefix] <= 0.0005 for prefix in ("if", "mf", "rf"))
            deepest = min(
                *distances.values(),
                table_distance,
                contact_hand_cube,
                contact_cross_finger,
            )
            if deepest < -PENETRATION_LIMIT:
                score -= 10000.0 + 1000.0 * (-deepest - PENETRATION_LIMIT) * 1000.0
            return {
                "vector": vector.copy(),
                "score": score,
                "distances": distances,
                "table_distance": table_distance,
                "contact_hand_cube": contact_hand_cube,
                "contact_cross_finger": contact_cross_finger,
            }

        for generation in range(GENERATIONS):
            if time.monotonic() - started >= TIME_BUDGET_S:
                break
            population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), lower, upper)
            population[0] = np.clip(mean, lower, upper)
            results = [evaluate(vector) for vector in population]
            results.sort(key=lambda result: float(result["score"]), reverse=True)
            history.extend(results[:20])
            elite_count = max(2, int(POPULATION * ELITE_FRACTION))
            elites = np.stack([result["vector"] for result in results[:elite_count]])
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), np.concatenate((np.full(16, 0.025), np.full(3, 0.002))))
            best = results[0]
            generation_lines.append(
                f"generation={generation + 1} best_score={best['score']:.6f} "
                f"distances_m={best['distances']} "
                f"contact_hand_cube_m={best['contact_hand_cube']:+.9f} "
                f"contact_cross_finger_m={best['contact_cross_finger']:+.9f} "
                f"cube_xyz={best['vector'][16:19].tolist()}"
            )

        history.sort(key=lambda result: float(result["score"]), reverse=True)
        static_rows = []
        seen = set()
        for result in history:
            vector = result["vector"]
            key = tuple(np.round(vector, 6))
            if key in seen:
                continue
            seen.add(key)
            for name, value in zip(HAND_JOINTS, vector[:16], strict=True):
                data.qpos[qpos_address(model, name)] = value
            data.qpos[cube_qpos : cube_qpos + 3] = vector[16:19]
            mujoco.mj_forward(model, data)
            contacts = []
            minimum_hand_cube = float("inf")
            for contact in data.contact[: data.ncon]:
                contact_geoms = {int(contact.geom1), int(contact.geom2)}
                if cube_geom in contact_geoms:
                    other = int(contact.geom2 if contact.geom1 == cube_geom else contact.geom1)
                    if other in geom_set:
                        contacts.append(geom_name(model, other))
                        minimum_hand_cube = min(minimum_hand_cube, float(contact.dist))
            prefixes = {finger_prefix(value) for value in contacts}
            fingers = sorted(prefix for prefix in prefixes if prefix in {"if", "mf", "rf"})
            thumb = "th" in prefixes
            minimum_table = min(
                float(mujoco.mj_geomDistance(model, data, table_geom, geom, 0.10, None))
                for geom in geoms
            )
            minimum_fingers = min(
                float(mujoco.mj_geomDistance(model, data, first, second, 0.01, None))
                for first, second in pairs
            )
            safe = min(minimum_hand_cube, minimum_table, minimum_fingers) >= -PENETRATION_LIMIT
            static_rows.append({
                **result,
                "contacts": sorted(set(contacts)),
                "thumb": thumb,
                "fingers": fingers,
                "minimum_hand_cube": minimum_hand_cube,
                "minimum_table": minimum_table,
                "minimum_fingers": minimum_fingers,
                "static_gate": thumb and len(fingers) >= 2 and safe,
            })
            if sum(row["static_gate"] for row in static_rows) >= 5:
                break

    passing = [row for row in static_rows if row["static_gate"]]
    pose_paths = []
    for index, row in enumerate(passing[:5], start=1):
        vector = row["vector"]
        path = ROOT / "poses" / f"grasp_optimizer_C{index:02d}.json"
        payload = {
            "test_only": True,
            "label": f"grasp_optimizer_C{index:02d}",
            "cube_start_xyz": vector[16:19].tolist(),
            "cube_start_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "mount": MOUNT,
            "joints": dict(zip(HAND_JOINTS, vector[:16].tolist(), strict=True)),
            "static_gate": {
                "thumb": row["thumb"],
                "non_thumb_fingers": row["fingers"],
                "contacts": row["contacts"],
                "minimum_hand_cube_m": row["minimum_hand_cube"],
                "minimum_hand_table_m": row["minimum_table"],
                "minimum_finger_finger_m": row["minimum_fingers"],
                "pass": True,
            },
            "optimizer": {
                "kind": "static grasp CEM-style sampling (not MPC, not learned)",
                "seed": SEED,
                "population": POPULATION,
                "generations": GENERATIONS,
                "elite_fraction": ELITE_FRACTION,
                "time_budget_s": TIME_BUDGET_S,
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        pose_paths.append(path)

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during grasp planning")
    lines = [
        "test=Prompt 12 LEAP right-hand static grasp optimizer",
        "method=project-native static grasp sampling; no Astra code/data copied",
        f"seed={SEED}",
        f"population={POPULATION}",
        f"generations_requested={GENERATIONS}",
        f"generations_completed={len(generation_lines)}",
        f"elite_fraction={ELITE_FRACTION}",
        f"time_budget_s={TIME_BUDGET_S}",
        f"elapsed_s={time.monotonic() - started:.3f}",
        "variables=16 LEAP joints + cube XYZ; mount and cube quaternion fixed",
        "objective_revision=contact-derived hand-cube and cross-finger penetration penalties enabled",
        "cube_physics=unchanged main-scene 0.05 m side, 0.125 kg, friction 0.2",
        f"static_candidates={len(passing)}",
        f"pose_files={[str(path) for path in pose_paths] or 'none'}",
        "dynamic_steps_run=0",
        "rotation_run=NO",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
        "",
        "generation_summary:",
        *generation_lines,
        "",
        "top_full_gate_checks:",
        "rank | score | thumb | fingers | hand_cube_m | hand_table_m | finger_finger_m | gate | cube_xyz",
    ]
    lines.extend(
        f"{index} | {row['score']:.6f} | {row['thumb']} | {row['fingers']} | "
        f"{row['minimum_hand_cube']:+.9f} | {row['minimum_table']:+.9f} | "
        f"{row['minimum_fingers']:+.9f} | {'PASS' if row['static_gate'] else 'FAIL'} | "
        f"{row['vector'][16:19].tolist()}"
        for index, row in enumerate(static_rows[:30], start=1)
    )
    lines.extend(("", "top_rejected_vectors:"))
    lines.extend(
        f"rank={index} vector={row['vector'].tolist()}"
        for index, row in enumerate(static_rows[:10], start=1)
    )
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: lines.index("generation_summary:")]))
    print(f"best_score={static_rows[0]['score']:.6f} best_thumb={static_rows[0]['thumb']} best_fingers={static_rows[0]['fingers']}")


if __name__ == "__main__":
    main()
