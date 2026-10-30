"""Validate the selected 15--20 degree KF03 twice from a clean state."""

from __future__ import annotations

import json

import mujoco

from search_articulated_face_followthrough import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    RING_JOINTS,
    SCENE,
)
from search_articulated_face_kf03 import OUTPUT_POSE, evaluate, replay_to_kf02
from validate_poses import ROOT, object_id, qpos_address, sha256


OUTPUT_LOG = ROOT / "logs" / "prompt26_face_kf03_validation.txt"


def run_once(model: mujoco.MjModel, candidate: dict[str, object]) -> dict[str, object]:
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    start_qpos = {
        name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS
    }
    baseline_angle = float(data.qpos[face_qpos])
    baseline_relative = data.xpos[cube_body] - data.xpos[palm_body]
    targets = {name: float(value) for name, value in candidate["target_joints"].items()}
    return evaluate(
        model,
        data,
        targets,
        start_qpos,
        baseline_angle,
        baseline_relative,
    )


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before KF03 validation")
    candidate = json.loads(OUTPUT_POSE.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    runs = [run_once(model, candidate), run_once(model, candidate)]
    repeatable = all(bool(run["pass"]) for run in runs)
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during KF03 validation")
    lines = [
        "test=Prompt 26 clean-state articulated-face KF03 validation",
        f"candidate={OUTPUT_POSE}",
        f"run_1={runs[0]}",
        f"run_2={runs[1]}",
        f"REPEATABLE_KF03_GATE={'PASS' if repeatable else 'FAIL'}",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
