"""Plan a turn-ready contact mode from the validated lifted R01 grasp."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from validate_poses import (
    EXPECTED_SCENE_SHA256,
    HAND_JOINTS,
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


SOURCE_POSE = ROOT / "poses" / "grasp_hold_refined_R01.json"
TURN_SEED = ROOT / "poses" / "turn_keyframes_K01.json"
OUTPUT_LOG = ROOT / "logs" / "prompt14_turn_ready_planner.txt"
SEED = 14001
POPULATION = 64
GENERATIONS = 8
ELITE_COUNT = 8
TIME_BUDGET_S = 45.0
TRANSITION_STEPS = 300
SCREEN_HOLD_STEPS = 100
SIDE_MARGIN_M = 0.005
NON_THUMB_JOINTS = tuple(name for name in HAND_JOINTS if not name.startswith("th_"))


def distal(name: str) -> bool:
    return "_md_" in name or "_ds_" in name or name.endswith("_tip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-side", action="store_true")
    args = parser.parse_args()
    required_side_fingers = 2 if args.dual_side else 1
    seed = 15003 if args.dual_side else SEED
    output_log = (
        ROOT / "logs" / "prompt15_dual_side_planner.txt"
        if args.dual_side else OUTPUT_LOG
    )
    pose_prefix = "dual_side_D" if args.dual_side else "turn_ready_T"
    started = time.monotonic()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    source_pose = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    turn_seed = json.loads(TURN_SEED.read_text(encoding="utf-8"))["keyframes"]["ring_guard"]
    rng = np.random.default_rng(seed)
    history: list[dict[str, object]] = []
    generation_lines: list[str] = []

    with TemporaryDirectory(prefix="prompt14_plan_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(MAIN_SCENE.read_text(encoding="utf-8"), Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, source_joints, _, _ = load_pose(SOURCE_POSE, model)
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
            *source_pose["cube_start_xyz"],
            *source_pose["cube_start_quat_wxyz"],
        )
        command = mount | source_joints
        for name, value in command.items():
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
        base_qpos = data.qpos.copy()
        base_qvel = data.qvel.copy()
        base_act = data.act.copy()
        base_ctrl = data.ctrl.copy()
        base_time = data.time
        base_relative = data.xpos[cube_body] - data.xpos[palm_body]

        mean_values = source_joints.copy()
        mean_values.update(turn_seed)
        if args.dual_side:
            index_seed = json.loads(
                (ROOT / "poses" / "turn_ready_T04_index_tip.json").read_text(encoding="utf-8")
            )["target_joints"]
            mean_values.update({name: index_seed[name] for name in NON_THUMB_JOINTS if name.startswith("if_")})
        mean = np.array([mean_values[name] for name in NON_THUMB_JOINTS], dtype=float)
        joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in NON_THUMB_JOINTS]
        lower = np.array([model.jnt_range[joint, 0] for joint in joint_ids])
        upper = np.array([model.jnt_range[joint, 1] for joint in joint_ids])
        std = np.full(len(mean), 0.22)

        def evaluate(vector: np.ndarray) -> dict[str, object]:
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.act[:] = base_act
            data.ctrl[:] = base_ctrl
            data.time = base_time
            mujoco.mj_forward(model, data)
            target = dict(zip(NON_THUMB_JOINTS, vector.tolist(), strict=True))
            deepest_cube = float("inf")
            deepest_self = float("inf")
            deepest_table = float("inf")
            maximum_drift = 0.0
            thumb_tail: list[bool] = []
            two_finger_tail: list[bool] = []
            distal_side_tail: list[bool] = []
            final_contacts: set[str] = set()
            final_side_contacts: set[str] = set()
            failed = False

            for step in range(TRANSITION_STEPS + SCREEN_HOLD_STEPS):
                if step < TRANSITION_STEPS:
                    phase = (step + 1) / TRANSITION_STEPS
                    blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                    for name, target_value in target.items():
                        start_value = source_joints[name]
                        data.ctrl[actuator_id(model, name)] = start_value + blend * (target_value - start_value)
                mujoco.mj_step(model, data)
                contacts: set[str] = set()
                side_contacts: set[str] = set()
                cube_z = float(data.xpos[cube_body, 2])
                for contact in data.contact[: data.ncon]:
                    first = int(contact.geom1)
                    second = int(contact.geom2)
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
                        first_prefix = finger_prefix(geom_name(model, first))
                        second_prefix = finger_prefix(geom_name(model, second))
                        if (
                            first_prefix in {"if", "mf", "rf", "th"}
                            and second_prefix in {"if", "mf", "rf", "th"}
                            and first_prefix != second_prefix
                        ):
                            deepest_self = min(deepest_self, float(contact.dist))
                prefixes = {finger_prefix(name) for name in contacts}
                thumb_tail.append("th" in prefixes)
                two_finger_tail.append(len(prefixes & {"if", "mf", "rf"}) >= 2)
                side_prefixes = {finger_prefix(name) for name in side_contacts}
                distal_side_tail.append(len(side_prefixes) >= required_side_fingers)
                if len(thumb_tail) > 20:
                    thumb_tail.pop(0)
                    two_finger_tail.pop(0)
                    distal_side_tail.pop(0)
                final_contacts = contacts
                final_side_contacts = side_contacts
                relative = data.xpos[cube_body] - data.xpos[palm_body]
                maximum_drift = max(maximum_drift, float(np.linalg.norm(relative - base_relative)))
                if (
                    min(deepest_cube, deepest_self, deepest_table) < -PENETRATION_LIMIT
                    or maximum_drift >= 0.015
                    or float(data.xpos[cube_body, 2]) < 0.105
                ):
                    failed = True
                    break

            prefixes = {finger_prefix(name) for name in final_contacts}
            fingers = sorted(prefixes & {"if", "mf", "rf"})
            thumb_fraction = float(np.mean(thumb_tail))
            support_fraction = float(np.mean(two_finger_tail))
            side_fraction = float(np.mean(distal_side_tail))
            distal_geoms = [
                geom for geom in hand
                if finger_prefix(geom_name(model, geom)) in {"if", "mf", "rf"}
                and distal(geom_name(model, geom))
            ]
            minimum_distal_distance = min(
                float(mujoco.mj_geomDistance(model, data, cube_geom, geom, 0.10, None))
                for geom in distal_geoms
            )
            deepest = min(deepest_cube, deepest_self, deepest_table)
            passed = (
                not failed
                and len(thumb_tail) == 20
                and thumb_fraction == 1.0
                and support_fraction == 1.0
                and side_fraction == 1.0
                and len(fingers) >= 2
                and deepest >= -PENETRATION_LIMIT
                and maximum_drift < 0.015
            )
            score = 40.0 * thumb_fraction + 30.0 * support_fraction + 50.0 * side_fraction
            score += 8.0 * len(fingers) + 8.0 * ("rf" in fingers)
            score -= 0.7 * maximum_drift * 1000.0
            score -= 0.4 * max(minimum_distal_distance, 0.0) * 1000.0
            if failed:
                score -= 10000.0
            return {
                "vector": vector.copy(),
                "score": score,
                "thumb_fraction": thumb_fraction,
                "support_fraction": support_fraction,
                "side_fraction": side_fraction,
                "fingers": fingers,
                "contacts": sorted(final_contacts),
                "side_contacts": sorted(final_side_contacts),
                "maximum_drift_m": maximum_drift,
                "minimum_distal_distance_m": minimum_distal_distance,
                "deepest_cube_m": deepest_cube,
                "deepest_self_m": deepest_self,
                "deepest_table_m": deepest_table,
                "screen_pass": passed,
            }

        for generation in range(GENERATIONS):
            if time.monotonic() - started >= TIME_BUDGET_S:
                break
            population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), lower, upper)
            population[0] = np.clip(mean, lower, upper)
            results = [evaluate(vector) for vector in population]
            results.sort(key=lambda row: float(row["score"]), reverse=True)
            history.extend(results[:12])
            elites = np.stack([row["vector"] for row in results[:ELITE_COUNT]])
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), np.full(len(mean), 0.02))
            best = results[0]
            generation_lines.append(
                f"generation={generation + 1} score={best['score']:.6f} "
                f"thumb={best['thumb_fraction']:.2f} supports={best['support_fraction']:.2f} "
                f"side={best['side_fraction']:.2f} fingers={best['fingers']} "
                f"side_contacts={best['side_contacts']} drift_mm={best['maximum_drift_m'] * 1000.0:.3f} "
                f"self_m={best['deepest_self_m']:+.6f} pass={best['screen_pass']}"
            )

    history.sort(key=lambda row: float(row["score"]), reverse=True)
    passing: list[dict[str, object]] = []
    seen: set[tuple[float, ...]] = set()
    for row in history:
        key = tuple(np.round(row["vector"], 6))
        if row["screen_pass"] and key not in seen:
            seen.add(key)
            passing.append(row)
        if len(passing) == 3:
            break
    pose_paths: list[Path] = []
    for index, row in enumerate(passing, start=1):
        target_joints = source_joints.copy()
        target_joints.update(dict(zip(NON_THUMB_JOINTS, row["vector"].tolist(), strict=True)))
        path = ROOT / "poses" / f"{pose_prefix}{index:02d}.json"
        payload = {
            "test_only": True,
            "label": path.stem,
            "source_held_pose": str(SOURCE_POSE),
            "transition_steps": TRANSITION_STEPS,
            "mount_lift_m": 0.030,
            "target_joints": target_joints,
            "screen": {name: row[name] for name in row if name != "vector"},
            "planner": {
                "kind": "turn-ready grasp CEM-style sampling (not MPC, not learned)",
                "seed": seed,
                "population": POPULATION,
                "generations": GENERATIONS,
                "elite_count": ELITE_COUNT,
                "time_budget_s": TIME_BUDGET_S,
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        pose_paths.append(path)

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during Prompt 14 planning")
    lines = [
        f"test={'Prompt 15 dual-side turn-ready planner' if args.dual_side else 'Prompt 14 turn-ready grasp planner'}",
        f"source_pose={SOURCE_POSE}",
        f"seed={seed}",
        f"population={POPULATION}",
        f"generations_requested={GENERATIONS}",
        f"generations_completed={len(generation_lines)}",
        f"elite_count={ELITE_COUNT}",
        f"time_budget_s={TIME_BUDGET_S}",
        f"elapsed_s={time.monotonic() - started:.3f}",
        f"transition_steps={TRANSITION_STEPS}",
        f"screen_hold_steps={SCREEN_HOLD_STEPS}",
        f"side_margin_m={SIDE_MARGIN_M}",
        f"required_distal_side_fingers={required_side_fingers}",
        f"screen_pass_candidates={len(passing)}",
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
    output_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
