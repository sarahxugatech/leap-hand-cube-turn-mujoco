"""Jointly refine Q01 support margin and target-axis torque in a TEST scene."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from run_scripted_turn import relative_angles_degrees
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


SOURCE_POSE = ROOT / "poses" / "fixed_base_axis_Q01.json"
OUTPUT_LOG = ROOT / "logs" / "prompt20_axis_support_refinement.txt"
SEED = 20001
POPULATION = 48
GENERATIONS = 10
ELITE_COUNT = 6
TIME_BUDGET_S = 45.0
SETTLE_STEPS = 200
LIFT_STEPS = 300
POST_LIFT_STEPS = 80
RAMP_STEPS = 160
HOLD_STEPS = 80
SUPPORT_GRACE = 12


def main() -> None:
    started = time.monotonic()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(SOURCE_POSE.read_text())
    probe_delta = float(payload["target_axis_probe"]["delta_rad"])
    rng = np.random.default_rng(SEED)
    history: list[dict[str, object]] = []
    generation_lines: list[str] = []

    with TemporaryDirectory(prefix="prompt20_refine_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(
            MAIN_SCENE.read_text(), Path(temporary), enable_yaw=True,
            mount_quat=payload["mount_quat_wxyz"],
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, source_joints, yaw_degrees, _ = load_pose(SOURCE_POSE, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        hand = hand_geoms(model)
        hand_set = set(hand)
        joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HAND_JOINTS]
        joint_low = np.array([model.jnt_range[joint, 0] for joint in joint_ids])
        joint_high = np.array([model.jnt_range[joint, 1] for joint in joint_ids])
        finger_pairs = [
            (a, b) for index, a in enumerate(hand) for b in hand[index + 1 :]
            if finger_prefix(geom_name(model, a)) in {"if", "mf", "rf", "th"}
            and finger_prefix(geom_name(model, b)) in {"if", "mf", "rf", "th"}
            and finger_prefix(geom_name(model, a)) != finger_prefix(geom_name(model, b))
        ]

        for name, value in mount.items() | source_joints.items():
            data.qpos[qpos_address(model, name)] = value
        data.qpos[qpos_address(model, "mount_yaw")] = math.radians(yaw_degrees)
        data.qpos[cube_qpos : cube_qpos + 7] = (
            *payload["cube_start_xyz"], *payload["cube_start_quat_wxyz"],
        )
        mujoco.mj_forward(model, data)
        palm_position = data.xpos[palm_body].copy()
        palm_rotation = data.xmat[palm_body].reshape(3, 3).copy()
        relative_position = palm_rotation.T @ (np.asarray(payload["cube_start_xyz"]) - palm_position)

        mean = np.array([source_joints[name] for name in HAND_JOINTS] + [0.0, 0.0, 0.0])
        std = np.array([0.04] * 16 + [0.0015] * 3)
        lower = np.concatenate((joint_low, [-0.006] * 3))
        upper = np.concatenate((joint_high, [0.006] * 3))

        def evaluate(vector: np.ndarray) -> dict[str, object]:
            mujoco.mj_resetData(model, data)
            joints = dict(zip(HAND_JOINTS, vector[:16].tolist(), strict=True))
            for name, value in mount.items() | joints.items():
                data.qpos[qpos_address(model, name)] = value
                data.ctrl[actuator_id(model, name)] = value
            yaw = math.radians(yaw_degrees)
            data.qpos[qpos_address(model, "mount_yaw")] = yaw
            data.ctrl[actuator_id(model, "mount_yaw")] = yaw
            mujoco.mj_forward(model, data)
            cube_position = data.xpos[palm_body] + data.xmat[palm_body].reshape(3, 3) @ (
                relative_position + vector[16:19]
            )
            data.qpos[cube_qpos : cube_qpos + 7] = (
                *cube_position, *payload["cube_start_quat_wxyz"],
            )
            mujoco.mj_forward(model, data)

            start_z = float(data.xpos[cube_body, 2])
            initial_relative = data.xpos[cube_body] - data.xpos[palm_body]
            maximum_drift = 0.0
            deepest_cube = deepest_table = deepest_self = float("inf")
            unsupported = 0
            failure = "none"
            turn_start_quaternion: np.ndarray | None = None
            final_contacts: set[str] = set()
            total_steps = SETTLE_STEPS + LIFT_STEPS + POST_LIFT_STEPS + RAMP_STEPS + HOLD_STEPS

            for step in range(total_steps):
                if SETTLE_STEPS <= step < SETTLE_STEPS + LIFT_STEPS:
                    phase = (step - SETTLE_STEPS + 1) / LIFT_STEPS
                    blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                    data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
                turn_start = SETTLE_STEPS + LIFT_STEPS + POST_LIFT_STEPS
                if step == turn_start:
                    turn_start_quaternion = data.xquat[cube_body].copy()
                if turn_start <= step < turn_start + RAMP_STEPS:
                    phase = (step - turn_start + 1) / RAMP_STEPS
                    blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                    data.ctrl[actuator_id(model, "rf_rot")] = joints["rf_rot"] + probe_delta * blend
                mujoco.mj_step(model, data)
                relative = data.xpos[cube_body] - data.xpos[palm_body]
                maximum_drift = max(maximum_drift, float(np.linalg.norm(relative - initial_relative)))
                contacts: set[str] = set()
                for contact in data.contact[: data.ncon]:
                    first, second = int(contact.geom1), int(contact.geom2)
                    pair = {first, second}
                    if cube_geom in pair:
                        other = second if first == cube_geom else first
                        if other in hand_set:
                            contacts.add(geom_name(model, other))
                            deepest_cube = min(deepest_cube, float(contact.dist))
                    elif table_geom in pair:
                        other = second if first == table_geom else first
                        if other in hand_set:
                            deepest_table = min(deepest_table, float(contact.dist))
                    elif first in hand_set and second in hand_set:
                        a, b = finger_prefix(geom_name(model, first)), finger_prefix(geom_name(model, second))
                        if a in {"if", "mf", "rf", "th"} and b in {"if", "mf", "rf", "th"} and a != b:
                            deepest_self = min(deepest_self, float(contact.dist))
                turn_active = step >= turn_start
                prefixes = {finger_prefix(name) for name in contacts}
                supported = "th" in prefixes and len(prefixes & {"if", "mf", "rf"}) >= 2
                unsupported = (0 if supported else unsupported + 1) if turn_active else 0
                final_contacts = contacts
                if turn_active and unsupported >= SUPPORT_GRACE:
                    failure = "support_loss"
                    break
                if maximum_drift >= 0.015:
                    failure = "drift_limit"
                    break
                if min(deepest_cube, deepest_table, deepest_self) < -PENETRATION_LIMIT:
                    failure = "penetration"
                    break

            cube_rise = float(data.xpos[cube_body, 2] - start_z)
            if turn_start_quaternion is None:
                yaw_rotation, tilt, geodesic = 0.0, 180.0, 180.0
            else:
                yaw_rotation, tilt, geodesic = relative_angles_degrees(
                    turn_start_quaternion, data.xquat[cube_body]
                )
            warnings = sum(int(data.warning[index].number) for index in range(mujoco.mjtWarning.mjNWARNING))
            if warnings and failure == "none":
                failure = "warning"
            axis_pass = abs(yaw_rotation) >= 1.0 and abs(yaw_rotation) > tilt
            gate = failure == "none" and cube_rise >= 0.020 and axis_pass
            rise_mm = cube_rise * 1000.0
            score = 15.0 * rise_mm + 12.0 * abs(yaw_rotation) - 10.0 * tilt - 0.5 * maximum_drift * 1000.0
            if not axis_pass:
                score -= 150.0
            if failure != "none":
                score -= 5000.0
                score -= 1000.0 * max(
                    0.0, -min(deepest_cube, deepest_table, deepest_self) - PENETRATION_LIMIT
                ) * 1000.0
            return {
                "vector": vector.copy(), "score": score, "cube_position": cube_position.copy(),
                "cube_rise": cube_rise, "yaw": yaw_rotation, "tilt": tilt,
                "geodesic": geodesic, "drift": maximum_drift, "failure": failure,
                "gate": gate, "contacts": sorted(final_contacts), "deepest_cube": deepest_cube,
                "deepest_table": deepest_table, "deepest_self": deepest_self,
            }

        baseline_mean = mean.copy()
        baseline = evaluate(mean)
        history.append(baseline)
        generation_lines.append(
            f"baseline score={baseline['score']:.6f} gate={baseline['gate']} "
            f"rise_mm={baseline['cube_rise'] * 1000:.3f} yaw={baseline['yaw']:+.3f} "
            f"tilt={baseline['tilt']:.3f} drift_mm={baseline['drift'] * 1000:.3f} "
            f"failure={baseline['failure']}"
        )
        for generation in range(GENERATIONS):
            if time.monotonic() - started >= TIME_BUDGET_S:
                break
            population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), lower, upper)
            population[0] = mean
            population[1] = baseline_mean
            results = [evaluate(vector) for vector in population]
            results.sort(key=lambda row: float(row["score"]), reverse=True)
            history.extend(results[:16])
            elites = np.stack([row["vector"] for row in results[:ELITE_COUNT]])
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), np.array([0.012] * 16 + [0.0004] * 3))
            best = results[0]
            generation_lines.append(
                f"generation={generation + 1} score={best['score']:.6f} gate={best['gate']} "
                f"rise_mm={best['cube_rise'] * 1000:.3f} yaw={best['yaw']:+.3f} "
                f"tilt={best['tilt']:.3f} drift_mm={best['drift'] * 1000:.3f} "
                f"failure={best['failure']}"
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
        if len(passing) == 3:
            break

    pose_paths: list[Path] = []
    for index, row in enumerate(passing, start=1):
        vector = row["vector"]
        path = ROOT / "poses" / f"axis_support_S{index:02d}.json"
        path.write_text(json.dumps({
            "test_only": True, "label": f"axis_support_S{index:02d}",
            "cube_start_xyz": row["cube_position"].tolist(),
            "cube_start_quat_wxyz": payload["cube_start_quat_wxyz"],
            "mount": payload["mount"], "mount_yaw_degrees": payload["mount_yaw_degrees"],
            "mount_quat_wxyz": payload["mount_quat_wxyz"],
            "joints": dict(zip(HAND_JOINTS, vector[:16].tolist(), strict=True)),
            "target_axis_probe": {"joint": "rf_rot", "delta_rad": probe_delta,
                                  "screen_cube_rise_m": row["cube_rise"],
                                  "screen_yaw_deg": row["yaw"], "screen_tilt_deg": row["tilt"],
                                  "screen_geodesic_deg": row["geodesic"],
                                  "screen_drift_m": row["drift"], "pass": True},
            "planner": {"kind": "joint axis/support CEM (not MPC, not learned)",
                        "seed": SEED, "population": POPULATION, "generations": GENERATIONS,
                        "elite_count": ELITE_COUNT, "time_budget_s": TIME_BUDGET_S},
        }, indent=2) + "\n")
        pose_paths.append(path)

    safe_axis_rows = [
        row for row in history
        if row["failure"] == "none"
        and abs(float(row["yaw"])) >= 1.0
        and abs(float(row["yaw"])) > float(row["tilt"])
    ]
    near = max(safe_axis_rows, key=lambda row: float(row["cube_rise"]), default=None)
    near_path: Path | None = None
    if not passing and near is not None:
        vector = near["vector"]
        near_path = ROOT / "poses" / "axis_support_near_N01.json"
        near_path.write_text(json.dumps({
            "test_only": True, "near_miss": True, "label": "axis_support_near_N01",
            "cube_start_xyz": near["cube_position"].tolist(),
            "cube_start_quat_wxyz": payload["cube_start_quat_wxyz"],
            "mount": payload["mount"], "mount_yaw_degrees": payload["mount_yaw_degrees"],
            "mount_quat_wxyz": payload["mount_quat_wxyz"],
            "joints": dict(zip(HAND_JOINTS, vector[:16].tolist(), strict=True)),
            "target_axis_probe": {"joint": "rf_rot", "delta_rad": probe_delta,
                                  "screen_cube_rise_m": near["cube_rise"],
                                  "screen_yaw_deg": near["yaw"], "screen_tilt_deg": near["tilt"],
                                  "screen_geodesic_deg": near["geodesic"],
                                  "screen_drift_m": near["drift"], "pass": False,
                                  "failed_gate": "cube rise < 0.020 m"},
            "planner": {"kind": "joint axis/support CEM (not MPC, not learned)",
                        "seed": SEED, "population": POPULATION, "generations": GENERATIONS,
                        "elite_count": ELITE_COUNT, "time_budget_s": TIME_BUDGET_S},
        }, indent=2) + "\n")

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during joint refinement")
    lines = [
        "test=Prompt 20 TEST-only joint support/axis refinement", f"seed={SEED}",
        f"population={POPULATION}", f"generations_requested={GENERATIONS}",
        f"generations_completed={len(generation_lines) - 1}", f"elapsed_s={time.monotonic() - started:.3f}",
        "variables=16 LEAP joints + palm-relative cube XYZ offset",
        "objective_revision=lift margin prioritized after first run peaked at 19.163 mm rise",
        "gates=cube rise >=20 mm; abs(world-Z yaw)>=1 deg; abs(yaw)>tilt; drift<15 mm; safe contacts",
        "full_90_degree_turn=NO", f"candidates={len(passing)}",
        f"pose_files={[str(path) for path in pose_paths] or 'none'}",
        f"near_miss_pose={near_path or 'none'}",
        f"near_miss_screen_rise_mm={near['cube_rise'] * 1000 if near else 'none'}",
        f"near_miss_screen_yaw_deg={near['yaw'] if near else 'none'}",
        f"near_miss_screen_tilt_deg={near['tilt'] if near else 'none'}",
        f"main_scene_sha256_before={before_hash}", f"main_scene_sha256_after={after_hash}", "",
        "generation_summary:", *generation_lines,
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("generation_summary:")]))


if __name__ == "__main__":
    main()
