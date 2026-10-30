"""Search a TEST-only O01 variant whose ring sweep produces world-Z yaw."""

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


SOURCE_POSE = ROOT / "poses" / "oriented_grasp_O01.json"
OUTPUT_LOG = ROOT / "logs" / "prompt18_target_axis_planner.txt"
SEED = 18001
POPULATION = 64
GENERATIONS = 8
ELITE_COUNT = 8
SETTLE_STEPS = 200
LIFT_STEPS = 300
RAMP_STEPS = 160
HOLD_STEPS = 80
TIME_BUDGET_S = 45.0
SUPPORT_GRACE = 12
RING_JOINTS = ("rf_mcp", "rf_rot", "rf_pip", "rf_dip")


def main() -> None:
    started = time.monotonic()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(SOURCE_POSE.read_text())
    rng = np.random.default_rng(SEED)
    history: list[dict[str, object]] = []
    generation_lines: list[str] = []

    with TemporaryDirectory(prefix="prompt18_axis_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(MAIN_SCENE.read_text(), Path(temporary), enable_yaw=True)
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, source_joints, yaw_degrees, _ = load_pose(SOURCE_POSE, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        table_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        hand = hand_geoms(model)
        hand_set = set(hand)
        finger_pairs = [
            (a, b) for index, a in enumerate(hand) for b in hand[index + 1 :]
            if finger_prefix(geom_name(model, a)) in {"if", "mf", "rf", "th"}
            and finger_prefix(geom_name(model, b)) in {"if", "mf", "rf", "th"}
            and finger_prefix(geom_name(model, a)) != finger_prefix(geom_name(model, b))
        ]
        ring_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RING_JOINTS]
        ring_low = np.array([model.jnt_range[joint, 0] for joint in ring_ids])
        ring_high = np.array([model.jnt_range[joint, 1] for joint in ring_ids])
        source_cube = np.asarray(payload["cube_start_xyz"], dtype=float)
        for name, value in mount.items() | source_joints.items():
            data.qpos[qpos_address(model, name)] = value
        data.qpos[qpos_address(model, "mount_yaw")] = math.radians(yaw_degrees)
        data.qpos[cube_qpos : cube_qpos + 7] = (*source_cube, *payload["cube_start_quat_wxyz"])
        mujoco.mj_forward(model, data)
        reference_palm_position = data.xpos[palm_body].copy()
        reference_palm_rotation = data.xmat[palm_body].reshape(3, 3).copy()
        reference_cube_rotation = np.empty(9)
        mujoco.mju_quat2Mat(reference_cube_rotation, np.asarray(payload["cube_start_quat_wxyz"]))
        reference_cube_rotation = reference_cube_rotation.reshape(3, 3)
        relative_cube_position = reference_palm_rotation.T @ (source_cube - reference_palm_position)
        relative_cube_rotation = reference_palm_rotation.T @ reference_cube_rotation
        mean = np.array([
            mount["mount_pitch"], mount["mount_roll"], math.radians(yaw_degrees),
            *(source_joints[name] for name in RING_JOINTS), 0.0, 0.0, 0.0, 0.10,
        ])
        std = np.array([0.04, 0.04, 0.06, 0.08, 0.12, 0.08, 0.08, 0.004, 0.004, 0.004, 0.04])
        lower = np.array([-0.35, -0.35, -0.523599, *ring_low, -0.015, -0.015, -0.015, 0.05])
        upper = np.array([0.35, 0.35, 0.523599, *ring_high, 0.015, 0.015, 0.015, 0.35])

        def evaluate(vector: np.ndarray) -> dict[str, object]:
            mujoco.mj_resetData(model, data)
            candidate_joints = source_joints | dict(zip(RING_JOINTS, vector[3:7].tolist(), strict=True))
            candidate_mount = mount | {"mount_pitch": float(vector[0]), "mount_roll": float(vector[1])}
            for name, value in candidate_mount.items() | candidate_joints.items():
                data.qpos[qpos_address(model, name)] = value
                data.ctrl[actuator_id(model, name)] = value
            data.qpos[qpos_address(model, "mount_yaw")] = vector[2]
            data.ctrl[actuator_id(model, "mount_yaw")] = vector[2]
            mujoco.mj_forward(model, data)
            candidate_palm_rotation = data.xmat[palm_body].reshape(3, 3)
            candidate_cube_position = (
                data.xpos[palm_body]
                + candidate_palm_rotation @ (relative_cube_position + vector[7:10])
            )
            candidate_cube_rotation = candidate_palm_rotation @ relative_cube_rotation
            candidate_cube_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(candidate_cube_quaternion, candidate_cube_rotation.ravel())
            data.qpos[cube_qpos : cube_qpos + 7] = (*candidate_cube_position, *candidate_cube_quaternion)
            mujoco.mj_forward(model, data)

            deepest_cube = deepest_table = deepest_self = float("inf")
            maximum_drift = 0.0
            unsupported = 0
            failed = False
            failure = "none"
            failure_step: int | None = None
            initial_relative = data.xpos[cube_body] - data.xpos[palm_body]
            turn_start_quaternion: np.ndarray | None = None
            final_contacts: set[str] = set()
            actual_amplitude = min(float(vector[10]), float(vector[4] - ring_low[1]))
            if actual_amplitude < 0.045:
                failed, failure = True, "rf_rot_range"

            for step in range(SETTLE_STEPS + LIFT_STEPS + RAMP_STEPS + HOLD_STEPS):
                if SETTLE_STEPS <= step < SETTLE_STEPS + LIFT_STEPS:
                    phase = (step - SETTLE_STEPS + 1) / LIFT_STEPS
                    blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                    data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
                if step == SETTLE_STEPS + LIFT_STEPS:
                    turn_start_quaternion = data.xquat[cube_body].copy()
                if SETTLE_STEPS + LIFT_STEPS <= step < SETTLE_STEPS + LIFT_STEPS + RAMP_STEPS:
                    phase = (step - SETTLE_STEPS - LIFT_STEPS + 1) / RAMP_STEPS
                    blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                    data.ctrl[actuator_id(model, "rf_rot")] = vector[4] - actual_amplitude * blend
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
                prefixes = {finger_prefix(name) for name in contacts}
                supported = "th" in prefixes and len(prefixes & {"if", "mf", "rf"}) >= 2
                turn_active = step >= SETTLE_STEPS + LIFT_STEPS
                unsupported = (0 if supported else unsupported + 1) if turn_active else 0
                final_contacts = contacts
                if turn_active and unsupported >= SUPPORT_GRACE:
                    failed, failure, failure_step = True, "support_loss", step
                    break
                if maximum_drift >= 0.015:
                    failed, failure, failure_step = True, "drift_limit", step
                    break
                if min(deepest_cube, deepest_table, deepest_self) < -PENETRATION_LIMIT:
                    failed, failure, failure_step = True, "penetration", step
                    break
            if turn_start_quaternion is None:
                yaw = 0.0
                tilt = geodesic = 180.0
            else:
                yaw, tilt, geodesic = relative_angles_degrees(turn_start_quaternion, data.xquat[cube_body])
            warnings = sum(int(data.warning[index].number) for index in range(mujoco.mjtWarning.mjNWARNING))
            if warnings:
                failed, failure = True, "warning"
            axis_pass = abs(yaw) >= 1.0 and abs(yaw) > tilt
            gate = not failed and axis_pass
            score = 18.0 * abs(yaw) - 12.0 * tilt - 120.0 * maximum_drift
            if failed:
                score -= 5000.0
                score -= 1000.0 * max(
                    0.0,
                    -min(deepest_cube, deepest_table, deepest_self) - PENETRATION_LIMIT,
                ) * 1000.0
            elif not axis_pass:
                score -= 250.0
            return {
                "vector": vector.copy(), "score": score, "yaw": yaw, "tilt": tilt,
                "geodesic": geodesic, "drift": maximum_drift, "amplitude": actual_amplitude,
                "contacts": sorted(final_contacts), "failure": failure, "gate": gate,
                "failure_step": failure_step,
                "deepest_cube": deepest_cube, "deepest_table": deepest_table,
                "deepest_self": deepest_self,
                "cube_position": candidate_cube_position.copy(),
                "cube_quaternion": candidate_cube_quaternion.copy(),
            }

        baseline_mean = mean.copy()
        baseline = evaluate(mean)
        generation_lines.append(
            f"baseline score={baseline['score']:.6f} gate={baseline['gate']} "
            f"yaw={baseline['yaw']:+.6f} tilt={baseline['tilt']:.6f} "
            f"failure={baseline['failure']} step={baseline['failure_step']} "
            f"deepest_cube={baseline['deepest_cube']:+.9f}"
        )
        grid_results: list[dict[str, object]] = [baseline]
        for pitch in np.linspace(-0.35, 0.35, 7):
            for roll in np.linspace(-0.35, 0.35, 7):
                vector = baseline_mean.copy()
                vector[0:2] = (pitch, roll)
                grid_results.append(evaluate(vector))
        grid_results.sort(key=lambda row: float(row["score"]), reverse=True)
        history.extend(grid_results)
        grid_best = grid_results[0]
        generation_lines.append(
            f"orientation_grid samples=50 score={grid_best['score']:.6f} gate={grid_best['gate']} "
            f"yaw={grid_best['yaw']:+.6f} tilt={grid_best['tilt']:.6f} "
            f"pitch={grid_best['vector'][0]:+.6f} roll={grid_best['vector'][1]:+.6f} "
            f"failure={grid_best['failure']}"
        )
        mean = grid_best["vector"].copy()
        std = np.array([0.04, 0.04, 0.04, 0.02, 0.03, 0.02, 0.02, 0.0005, 0.0005, 0.0005, 0.02])
        for generation in range(GENERATIONS):
            if time.monotonic() - started >= TIME_BUDGET_S:
                break
            population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), lower, upper)
            population[0] = np.clip(mean, lower, upper)
            population[1] = baseline_mean
            results = [evaluate(vector) for vector in population]
            results.sort(key=lambda row: float(row["score"]), reverse=True)
            history.extend(results[:16])
            elites = np.stack([row["vector"] for row in results[:ELITE_COUNT]])
            mean = np.mean(elites, axis=0)
            std = np.maximum(np.std(elites, axis=0), [0.015] * 3 + [0.025] * 4 + [0.0015] * 3 + [0.01])
            best = results[0]
            generation_lines.append(
                f"generation={generation + 1} score={best['score']:.6f} gate={best['gate']} "
                f"yaw={best['yaw']:+.6f} tilt={best['tilt']:.6f} drift_mm={best['drift'] * 1000:.3f} "
                f"amplitude={best['amplitude']:.6f} failure={best['failure']} step={best['failure_step']} "
                f"deepest_cube={best['deepest_cube']:+.9f} deepest_self={best['deepest_self']:+.9f} "
                f"deepest_table={best['deepest_table']:+.9f}"
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
        joints = source_joints | dict(zip(RING_JOINTS, vector[3:7].tolist(), strict=True))
        path = ROOT / "poses" / f"target_axis_A{index:02d}.json"
        path.write_text(json.dumps({
            "test_only": True, "label": f"target_axis_A{index:02d}",
            "cube_start_xyz": row["cube_position"].tolist(),
            "cube_start_quat_wxyz": row["cube_quaternion"].tolist(),
            "mount": mount | {"mount_pitch": float(vector[0]), "mount_roll": float(vector[1])},
            "mount_yaw_degrees": math.degrees(float(vector[2])), "joints": joints,
            "target_axis_probe": {
                "joint": "rf_rot", "delta_rad": -float(row["amplitude"]),
                "screen_yaw_deg": row["yaw"], "screen_tilt_deg": row["tilt"],
                "screen_geodesic_deg": row["geodesic"], "screen_drift_m": row["drift"],
                "contacts": row["contacts"], "pass": True,
            },
            "planner": {"kind": "dynamic target-axis CEM (not MPC, not learned)", "seed": SEED,
                        "population": POPULATION, "generations": GENERATIONS,
                        "elite_count": ELITE_COUNT, "time_budget_s": TIME_BUDGET_S},
        }, indent=2) + "\n")
        pose_paths.append(path)

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during target-axis planning")
    lines = [
        "test=Prompt 18 TEST-only target-axis dynamic planner", f"seed={SEED}",
        f"population={POPULATION}", f"generations_completed={GENERATIONS}",
        f"elapsed_s={time.monotonic() - started:.3f}",
        "variables=mount pitch/roll/yaw + 4 ring joints + palm-relative cube XYZ offset + rf_rot amplitude",
        "fixed=thumb/index/middle joints and cube physics", "full_90_degree_turn=NO",
        f"axis_gate=abs(world-Z yaw)>=1 deg and abs(yaw)>tilt",
        f"candidates={len(passing)}", f"pose_files={[str(path) for path in pose_paths] or 'none'}",
        f"main_scene_sha256_before={before_hash}", f"main_scene_sha256_after={after_hash}", "",
        "generation_summary:", *generation_lines,
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("generation_summary:")]))


if __name__ == "__main__":
    main()
