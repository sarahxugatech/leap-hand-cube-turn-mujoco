"""Check the TEST-only articulated face scene without grasping or turning it."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCENE = ROOT / "scenes" / "table_cube.xml"
TEST_SCENE = ROOT / "scenes" / "table_articulated_face_test.xml"
LOG = ROOT / "logs" / "prompt22_articulated_face_scene_check.txt"
EXPECTED_MAIN_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
STEPS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise KeyError(name)
    return value


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    if main_before != EXPECTED_MAIN_SHA256:
        raise RuntimeError(f"frozen main scene hash mismatch: {main_before}")
    model = mujoco.MjModel.from_xml_path(str(TEST_SCENE))
    if model.neq:
        raise RuntimeError("articulated TEST scene must not contain equality constraints")

    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    if model.jnt_type[cube_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("cube body is not free")
    if model.jnt_type[face_joint] != mujoco.mjtJoint.mjJNT_HINGE:
        raise RuntimeError("face layer is not hinged")

    data = mujoco.MjData(model)
    key = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "start")
    mujoco.mj_resetDataKeyframe(model, data, key)
    for name in ("mount_x", "mount_y", "mount_z", "mount_pitch", "mount_roll"):
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act")
        data.ctrl[actuator] = data.qpos[int(model.jnt_qposadr[joint])]
    mujoco.mj_forward(model, data)

    cube_start = data.xpos[cube_body].copy()
    face_qpos = int(model.jnt_qposadr[face_joint])
    face_start = float(data.qpos[face_qpos])
    maximum_cube_drift = 0.0
    maximum_face_drift = 0.0
    warnings: list[str] = []
    old_handler = mujoco.set_mju_user_warning(warnings.append)
    try:
        for _ in range(STEPS):
            mujoco.mj_step(model, data)
            maximum_cube_drift = max(
                maximum_cube_drift,
                float(np.linalg.norm(data.xpos[cube_body] - cube_start)),
            )
            maximum_face_drift = max(
                maximum_face_drift,
                abs(float(data.qpos[face_qpos]) - face_start),
            )
    finally:
        mujoco.set_mju_user_warning(old_handler)
    warnings.extend(
        mujoco.mjtWarning(index).name
        for index in range(mujoco.mjtWarning.mjNWARNING)
        if data.warning[index].number
    )

    cube_end = data.xpos[cube_body].copy()
    face_end = float(data.qpos[face_qpos])
    total_mass = float(model.body_mass[cube_body] + model.body_mass[face_body])
    face_range = model.jnt_range[face_joint].copy()
    passive_pass = (
        maximum_cube_drift < 0.002
        and maximum_face_drift < np.deg2rad(0.1)
        and not warnings
    )
    main_after = sha256(MAIN_SCENE)
    if main_after != main_before:
        raise RuntimeError("frozen main scene changed during articulated scene check")

    lines = [
        "test=Prompt 22 TEST-only articulated face scene check",
        f"test_scene={TEST_SCENE}",
        f"test_scene_sha256={sha256(TEST_SCENE)}",
        f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}",
        "cube_freejoint=YES",
        "face_joint=face_turn",
        f"face_axis={model.jnt_axis[face_joint].tolist()}",
        f"face_range_rad={face_range.tolist()}",
        f"face_range_deg={np.rad2deg(face_range).tolist()}",
        f"face_damping={model.dof_damping[int(model.jnt_dofadr[face_joint])]:.9f}",
        f"face_frictionloss={model.dof_frictionloss[int(model.jnt_dofadr[face_joint])]:.9f}",
        f"cube_core_mass_kg={model.body_mass[cube_body]:.9f}",
        f"face_layer_mass_kg={model.body_mass[face_body]:.9f}",
        f"total_cube_mass_kg={total_mass:.9f}",
        f"cube_start_m={cube_start.tolist()}",
        f"cube_end_m={cube_end.tolist()}",
        f"maximum_cube_drift_m={maximum_cube_drift:.9f}",
        f"face_start_rad={face_start:+.9f}",
        f"face_end_rad={face_end:+.9f}",
        f"maximum_face_drift_rad={maximum_face_drift:.9f}",
        f"warnings={warnings or 'none'}",
        f"PASSIVE_STABILITY={'PASS' if passive_pass else 'FAIL'}",
        "grasp_attempted=NO",
        "face_turn_attempted=NO",
        "cube_hand_attachment=NO",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
    ]
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
