"""Check whether the true ring tip can reach the movable face naturally."""

from __future__ import annotations

import json
import time

import mujoco
import numpy as np

from plan_initial_ring_core_support import replay_lifted_grasp
from search_articulated_face_followthrough import MAIN_SCENE, SCENE, SOURCE_POSE
from search_global_redundant_grasp import EXPECTED_MAIN_SHA256, EXPECTED_SCENE_SHA256
from validate_poses import (
    PENETRATION_LIMIT,
    ROOT,
    finger_prefix,
    geom_name,
    hand_geoms,
    load_pose,
    object_id,
    qpos_address,
    sha256,
)


OUTPUT_LOG = ROOT / "logs" / "prompt41_natural_ring_tip_face_search.txt"
OUTPUT_POSE = ROOT / "poses" / "natural_ring_tip_face_R01.json"
SEED = 41_001
SAMPLES = 100_000


def main() -> None:
    if sha256(MAIN_SCENE) != EXPECTED_MAIN_SHA256 or sha256(SCENE) != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    replay_lifted_grasp(model, data)
    _, source_joints, _, _ = load_pose(SOURCE_POSE, model)
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    tip_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_tip")
    hand_set = set(hand_geoms(model))
    base_qpos = data.qpos.copy()
    rng = np.random.default_rng(SEED)
    best_distance = float("inf")
    best_side_distance = float("inf")
    safe_side_candidates: list[dict[str, object]] = []
    penetration_rejects = 0
    support_rejects = 0
    started = time.monotonic()

    for _ in range(SAMPLES):
        pip = float(rng.uniform(-0.1, 1.5))
        values = {
            "rf_mcp": float(rng.uniform(0.0, 1.8)),
            "rf_rot": float(rng.uniform(-0.5, 0.5)),
            "rf_pip": pip,
            "rf_dip": float(rng.uniform(max(-0.1, pip - 0.7), min(1.6, pip + 0.7))),
        }
        data.qpos[:] = base_qpos
        data.qvel[:] = 0.0
        for name, value in values.items():
            data.qpos[qpos_address(model, name)] = value
        mujoco.mj_forward(model, data)
        fromto = np.empty(6)
        distance = float(mujoco.mj_geomDistance(model, data, face_geom, tip_geom, 0.25, fromto))
        best_distance = min(best_distance, distance)
        local = data.xmat[face_body].reshape(3, 3).T @ (fromto[:3] - data.xpos[face_body])
        on_side = max(abs(local[0]), abs(local[1])) >= 0.023 and abs(local[2]) <= 0.009
        if on_side:
            best_side_distance = min(best_side_distance, distance)
        if not on_side or not -0.002 <= distance <= 0.0015:
            continue
        minimum_distance = min(
            (float(contact.dist) for contact in data.contact[: data.ncon]),
            default=float("inf"),
        )
        if minimum_distance < -PENETRATION_LIMIT:
            penetration_rejects += 1
            continue
        core_prefixes: set[str] = set()
        for contact in data.contact[: data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            if core_geom not in {first, second}:
                continue
            other = second if first == core_geom else first
            if other in hand_set:
                prefix = finger_prefix(geom_name(model, other))
                if prefix:
                    core_prefixes.add(prefix)
        if not {"th", "if", "mf"}.issubset(core_prefixes):
            support_rejects += 1
            continue
        safe_side_candidates.append(
            {
                "tip_face_distance_m": distance,
                "minimum_contact_distance_m": minimum_distance,
                "target_joints": source_joints | values,
            }
        )

    safe_side_candidates.sort(key=lambda item: abs(float(item["tip_face_distance_m"]) + 0.0002))
    if safe_side_candidates:
        OUTPUT_POSE.write_text(
            json.dumps(
                {
                    "test_only": True,
                    "label": "natural_ring_tip_face_R01",
                    "source_pose": str(SOURCE_POSE),
                    "natural_constraints": {
                        "rf_rot_abs_max_rad": 0.5,
                        "rf_mcp_rad": [0.0, 1.8],
                        "rf_pip_rad": [-0.1, 1.5],
                        "rf_dip_pip_difference_abs_max_rad": 0.7,
                    },
                    "static_result": safe_side_candidates[0],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    lines = [
        "test=natural true ring-tip movable-face reachability",
        f"seed={SEED}",
        f"samples={SAMPLES}",
        "target_geom=rf_tip",
        "rf_rot_range_rad=[-0.5,+0.5]",
        f"best_any_tip_face_distance_m={best_distance}",
        f"best_side_tip_face_distance_m={best_side_distance}",
        f"penetration_rejects={penetration_rejects}",
        f"support_rejects={support_rejects}",
        f"safe_side_candidates={len(safe_side_candidates)}",
        f"best_candidate={json.dumps(safe_side_candidates[0], sort_keys=True) if safe_side_candidates else 'NONE'}",
        f"elapsed_s={time.monotonic() - started:.6f}",
        "dynamic_validation=NOT_RUN" if not safe_side_candidates else "dynamic_validation=PENDING",
        "face_turn_attempted=NO",
        f"main_scene_sha256={sha256(MAIN_SCENE)}",
        f"test_scene_sha256={sha256(SCENE)}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
