"""Sample one LEAP finger-gait keyframe sequence after scripted feasibility."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from run_scripted_turn import POSE, relative_angles_degrees
from validate_poses import (
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
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
    test_scene_copy,
)


SEED = 13001
POPULATION = 24
GENERATIONS = 6
ELITE_COUNT = 5
TIME_BUDGET_S = 45.0
OUTPUT_LOG = ROOT / "logs" / "prompt13_turn_keyframe_planner.txt"
MEAN = np.array([1.20, -1.00, 0.60, 0.50, 0.70, -0.70, -0.10, 0.05])
STD = np.array([0.18, 0.10, 0.18, 0.15, 0.14, 0.14, 0.08, 0.08])
LOWER = np.array([0.60, -1.047, 0.20, 0.20, 0.30, -0.90, -0.20, -0.10])
UPPER = np.array([1.40, -0.60, 1.00, 0.80, 0.90, -0.30, 0.10, 0.20])


def main() -> None:
    started = time.monotonic()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(POSE.read_text(encoding="utf-8"))
    source = MAIN_SCENE.read_text(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    generation_lines: list[str] = []
    history: list[dict[str, object]] = []

    with TemporaryDirectory(prefix="turn_plan_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(source, Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, joints, _, _ = load_pose(POSE, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        hand = hand_geoms(model)
        hand_set = set(hand)

        data.qpos[cube_qpos : cube_qpos + 7] = (
            *payload["cube_start_xyz"],
            *payload["cube_start_quat_wxyz"],
        )
        base_command = mount | joints
        for name, value in base_command.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        mujoco.mj_forward(model, data)
        for _ in range(500):
            mujoco.mj_step(model, data)
        for step in range(500):
            phase = min((step + 1) / 300, 1.0)
            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
            data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
            mujoco.mj_step(model, data)
        base_command["mount_z"] = mount["mount_z"] + 0.030
        base_qpos = data.qpos.copy()
        base_qvel = data.qvel.copy()
        base_act = data.act.copy()
        base_ctrl = data.ctrl.copy()
        base_time = data.time
        turn_start_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
        relative_start = data.xpos[cube_body] - data.xpos[palm_body]

        def evaluate(vector: np.ndarray) -> dict[str, object]:
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.act[:] = base_act
            data.ctrl[:] = base_ctrl
            data.time = base_time
            mujoco.mj_forward(model, data)
            command = base_command.copy()
            deepest_cube = float("inf")
            deepest_self = float("inf")
            deepest_table = float("inf")
            maximum_drift = 0.0
            minimum_z = float("inf")
            final_prefixes: set[str] = set()
            thumb_tail: list[bool] = []

            def observe() -> None:
                nonlocal deepest_cube, deepest_self, deepest_table
                nonlocal maximum_drift, minimum_z, final_prefixes
                prefixes: set[str] = set()
                for contact in data.contact[: data.ncon]:
                    first = int(contact.geom1)
                    second = int(contact.geom2)
                    pair = {first, second}
                    if cube_geom in pair:
                        other = second if first == cube_geom else first
                        if other in hand_set:
                            prefixes.add(finger_prefix(geom_name(model, other)))
                            deepest_cube = min(deepest_cube, float(contact.dist))
                    elif table_geom in pair:
                        other = second if first == table_geom else first
                        if other in hand_set:
                            deepest_table = min(deepest_table, float(contact.dist))
                    elif first in hand_set and second in hand_set:
                        first_prefix = finger_prefix(geom_name(model, first))
                        second_prefix = finger_prefix(geom_name(model, second))
                        if (
                            first_prefix in {"if", "mf", "rf", "th"}
                            and second_prefix in {"if", "mf", "rf", "th"}
                            and first_prefix != second_prefix
                        ):
                            deepest_self = min(deepest_self, float(contact.dist))
                final_prefixes = prefixes
                thumb_tail.append("th" in prefixes)
                if len(thumb_tail) > 100:
                    thumb_tail.pop(0)
                relative = data.xpos[cube_body] - data.xpos[palm_body]
                maximum_drift = max(maximum_drift, float(np.linalg.norm(relative - relative_start)))
                minimum_z = min(minimum_z, float(data.xpos[cube_body, 2]))

            def phase(target: dict[str, float], steps: int) -> None:
                start = command.copy()
                for local_step in range(steps):
                    amount = (local_step + 1) / steps
                    blend = 0.5 - 0.5 * math.cos(math.pi * amount)
                    for name, target_value in target.items():
                        value = start[name] + blend * (target_value - start[name])
                        data.ctrl[actuator_id(model, name)] = value
                        command[name] = value
                    mujoco.mj_step(model, data)
                    observe()

            rf_mcp, rf_rot, rf_pip, rf_dip, if_rot, mf_rot, if_release, mf_release = vector
            phase({"rf_mcp": rf_mcp, "rf_rot": rf_rot, "rf_pip": rf_pip, "rf_dip": rf_dip}, 300)
            phase({"if_rot": if_rot, "mf_rot": mf_rot}, 300)
            phase({"if_mcp": if_release, "mf_mcp": mf_release}, 100)
            phase({"if_rot": joints["if_rot"], "mf_rot": joints["mf_rot"]}, 180)
            phase({"if_mcp": joints["if_mcp"], "mf_mcp": joints["mf_mcp"]}, 140)
            phase({}, 200)
            end_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
            yaw, tilt, geodesic = relative_angles_degrees(turn_start_quat, end_quat)
            thumb_sustained = len(thumb_tail) == 100 and all(thumb_tail)
            fingers = sorted(final_prefixes & {"if", "mf", "rf"})
            deepest = min(deepest_cube, deepest_self, deepest_table)
            safe = deepest >= -PENETRATION_LIMIT and maximum_drift < 0.015 and minimum_z >= 0.105
            score = 10.0 * max(yaw, 0.0) - 2.0 * tilt - 200.0 * maximum_drift
            score += 20.0 * thumb_sustained + 10.0 * min(len(fingers), 2)
            if not safe:
                score -= 10000.0
                if deepest < -PENETRATION_LIMIT:
                    score -= 1000.0 * (-deepest - PENETRATION_LIMIT) * 1000.0
                if maximum_drift >= 0.015:
                    score -= 1000.0 * (maximum_drift - 0.015) * 1000.0
            return {
                "vector": vector.copy(),
                "score": score,
                "yaw_deg": yaw,
                "tilt_deg": tilt,
                "geodesic_deg": geodesic,
                "thumb_sustained": thumb_sustained,
                "fingers": fingers,
                "maximum_drift_m": maximum_drift,
                "deepest_cube_m": deepest_cube,
                "deepest_self_m": deepest_self,
                "deepest_table_m": deepest_table,
                "safe": safe,
            }

        mean = MEAN.copy()
        std = STD.copy()
        for generation in range(GENERATIONS):
            if time.monotonic() - started >= TIME_BUDGET_S:
                break
            population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), LOWER, UPPER)
            population[0] = np.clip(mean, LOWER, UPPER)
            results = [evaluate(vector) for vector in population]
            results.sort(key=lambda row: float(row["score"]), reverse=True)
            history.extend(results[:10])
            elites = np.stack([row["vector"] for row in results[:ELITE_COUNT]])
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), np.array([0.02] * 8))
            best = results[0]
            generation_lines.append(
                f"generation={generation + 1} score={best['score']:.6f} yaw_deg={best['yaw_deg']:+.3f} "
                f"tilt_deg={best['tilt_deg']:.3f} drift_mm={best['maximum_drift_m'] * 1000.0:.3f} "
                f"self_m={best['deepest_self_m']:+.6f} thumb={best['thumb_sustained']} "
                f"fingers={best['fingers']} safe={best['safe']}"
            )

    history.sort(key=lambda row: float(row["score"]), reverse=True)
    safe_rows = [row for row in history if row["safe"] and row["thumb_sustained"] and len(row["fingers"]) >= 2]
    pose_paths: list[Path] = []
    seen: set[tuple[float, ...]] = set()
    for row in safe_rows:
        key = tuple(np.round(row["vector"], 6))
        if key in seen:
            continue
        seen.add(key)
        path = ROOT / "poses" / f"turn_keyframes_K{len(pose_paths) + 1:02d}.json"
        vector = row["vector"]
        payload_out = {
            "test_only": True,
            "source_held_pose": str(POSE),
            "target_rotation_deg": 90.0,
            "tolerance_deg": 10.0,
            "keyframes": {
                "ring_guard": dict(zip(("rf_mcp", "rf_rot", "rf_pip", "rf_dip"), vector[:4].tolist(), strict=True)),
                "differential_push": {"if_rot": float(vector[4]), "mf_rot": float(vector[5])},
                "release": {"if_mcp": float(vector[6]), "mf_mcp": float(vector[7])},
            },
            "screen": {name: row[name] for name in row if name not in {"vector"}},
            "planner": {
                "kind": "sampled keyframe planner (not MPC, not learned)",
                "seed": SEED,
                "population": POPULATION,
                "generations": GENERATIONS,
                "elite_count": ELITE_COUNT,
                "time_budget_s": TIME_BUDGET_S,
            },
        }
        path.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")
        pose_paths.append(path)
        if len(pose_paths) == 3:
            break

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during turn planning")
    lines = [
        "test=Prompt 13 sampled turn-keyframe planner",
        "precondition=scripted run02 produced +5.348 deg measured Z rotation",
        f"seed={SEED}",
        f"population={POPULATION}",
        f"generations_requested={GENERATIONS}",
        f"generations_completed={len(generation_lines)}",
        f"elite_count={ELITE_COUNT}",
        f"time_budget_s={TIME_BUDGET_S}",
        f"elapsed_s={time.monotonic() - started:.3f}",
        f"safe_held_candidates={len(safe_rows)}",
        f"pose_files={[str(path) for path in pose_paths] or 'none'}",
        "MPC=NO",
        "learned_policy=NO",
        "cube_hand_attachment=NO",
        "cube_qpos_updates_during_run=0",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
        "",
        *generation_lines,
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
