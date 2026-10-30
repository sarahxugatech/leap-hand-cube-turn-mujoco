"""Verify the TEST-only 2x2 cube geometry, mass, and face hinge."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "table_2x2_cube_test.xml"
LOG = ROOT / "logs" / "prompt35_2x2_cube_scene_check.txt"


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise KeyError(name)
    return value


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "start")
    cube = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    layer = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "face_layer")
    hinge = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    cubies = [
        object_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"cubie_{name}")
        for name in ("dpp", "dpn", "dnp", "dnn", "upp", "upn", "unp", "unn")
    ]
    mujoco.mj_resetDataKeyframe(model, data, key)
    for name in ("mount_x", "mount_y", "mount_z", "mount_pitch", "mount_roll"):
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act")
        data.ctrl[actuator] = data.qpos[int(model.jnt_qposadr[joint])]
    mujoco.mj_forward(model, data)

    total_mass = float(model.body_mass[cube] + model.body_mass[layer])
    visual_geoms_are_passive = all(
        model.geom_contype[index] == 0 and model.geom_conaffinity[index] == 0
        for index in cubies
    )
    hinge_qpos = int(model.jnt_qposadr[hinge])
    data.qpos[hinge_qpos] = math.pi / 2
    mujoco.mj_forward(model, data)
    upper_positions = np.array([data.geom_xpos[index] for index in cubies[4:]])
    expected_xy = {
        (-0.0125, -0.0125),
        (-0.0125, 0.0125),
        (0.0125, -0.0125),
        (0.0125, 0.0125),
    }
    measured_xy = {
        tuple(np.round(position[:2], 4)) for position in upper_positions
    }
    checks = {
        "eight_visual_cubies": len(cubies) == 8,
        "total_mass_0p125kg": abs(total_mass - 0.125) < 1e-9,
        "two_equal_mass_layers": abs(float(model.body_mass[cube]) - 0.0625) < 1e-9
        and abs(float(model.body_mass[layer]) - 0.0625) < 1e-9,
        "visuals_are_noncolliding": visual_geoms_are_passive,
        "face_hinge_90deg": abs(float(data.qpos[hinge_qpos]) - math.pi / 2) < 1e-12,
        "upper_layer_rotates_as_four": measured_xy == expected_xy,
        "no_equality_attachment": model.neq == 0,
    }
    passed = all(checks.values())
    lines = [
        "test=2x2 cube TEST scene load and geometry check",
        f"scene={SCENE}",
        f"scene_sha256={hashlib.sha256(SCENE.read_bytes()).hexdigest()}",
        f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}",
        f"cube_core_mass_kg={model.body_mass[cube]:.9f}",
        f"top_layer_mass_kg={model.body_mass[layer]:.9f}",
        f"total_cube_mass_kg={total_mass:.9f}",
        f"face_axis={model.jnt_axis[hinge].tolist()}",
        f"face_range_deg={np.rad2deg(model.jnt_range[hinge]).tolist()}",
        f"upper_cubie_xy_after_90deg={sorted(measured_xy)}",
        *(f"{name}={'PASS' if value else 'FAIL'}" for name, value in checks.items()),
        f"SCENE_CHECK={'PASS' if passed else 'FAIL'}",
        "grasp_revalidated=NO",
        "old_trajectory_revalidated=NO",
    ]
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
