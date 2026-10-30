"""Forward-kinematics-only T3 fingertip-to-cube diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes/table_cube.xml"
POSE = ROOT / "poses/power_wrap_B_thumb_T3.json"
MOUNT_JOINTS = ("mount_x", "mount_y", "mount_z", "mount_pitch", "mount_roll")
HAND_JOINTS = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
HALF_SIZE = 0.025
FACES = {
    "-X": (0, -1), "+X": (0, 1),
    "-Y": (1, -1), "+Y": (1, 1),
    "bottom": (2, -1), "top": (2, 1),
}


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"missing model object: {name}")
    return result


def test_scene(directory: Path) -> Path:
    source = SCENE.read_text(encoding="utf-8")
    relative = 'file="../menagerie/leap_hand/right_hand.xml"'
    absolute = f'file="{(ROOT / "menagerie/leap_hand/right_hand.xml").resolve()}"'
    if source.count(relative) != 1:
        raise ValueError("main scene LEAP path is not unique")
    path = directory / "table_cube_thumb_distance.xml"
    path.write_text(source.replace(relative, absolute), encoding="utf-8")
    return path


def face_closest(point: np.ndarray, centre: np.ndarray, face: str) -> tuple[np.ndarray, np.ndarray, float]:
    axis, sign = FACES[face]
    local = point - centre
    closest = np.clip(local, -HALF_SIZE, HALF_SIZE)
    closest[axis] = sign * HALF_SIZE
    world = centre + closest
    delta = world - point
    return world, delta, float(np.linalg.norm(delta))


def vector(values: np.ndarray) -> str:
    return np.array2string(values, precision=9, separator=", ")


def main() -> None:
    pose = json.loads(POSE.read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="thumb_distance_", dir=ROOT / "logs") as temporary:
        model = mujoco.MjModel.from_xml_path(str(test_scene(Path(temporary))))
        data = mujoco.MjData(model)
        key = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "start")
        mujoco.mj_resetDataKeyframe(model, data, key)
        for name in MOUNT_JOINTS:
            joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = float(pose["mount"][name])
        for name in HAND_JOINTS:
            joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = float(pose["joints"][name])
        mujoco.mj_forward(model, data)

        cube = data.xpos[object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")].copy()
        points = {
            "thumb_tip": data.geom_xpos[object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "th_tip")].copy(),
            "index_tip": data.geom_xpos[object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "if_tip")].copy(),
            "middle_tip": data.geom_xpos[object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "mf_tip")].copy(),
            "ring_tip": data.geom_xpos[object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_tip")].copy(),
        }
        palm = data.xpos[object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")].copy()
        thumb_base = data.xpos[object_id(model, mujoco.mjtObj.mjOBJ_BODY, "th_bs")].copy()
        print(f"cube_centre={vector(cube)}")
        print(f"palm_centre={vector(palm)}")
        print(f"thumb_base={vector(thumb_base)}")
        for name, point in points.items():
            print(f"{name}={vector(point)}")
        thumb = points["thumb_tip"]
        for face in ("-X", "+X", "-Y", "+Y", "top"):
            closest, delta, distance = face_closest(thumb, cube, face)
            print(f"thumb_face_{face}: distance={distance:.9f} closest={vector(closest)} delta={vector(delta)}")
        candidates = {
            face: face_closest(thumb, cube, face) for face in FACES
        }
        closest_face, (closest_point, delta, distance) = min(
            candidates.items(), key=lambda item: item[1][2]
        )
        print(f"thumb_closest_face={closest_face}")
        print(f"thumb_closest_distance={distance:.9f}")
        print(f"thumb_closest_point={vector(closest_point)}")
        print(f"thumb_closest_delta_DX_DY_DZ={vector(delta)}")
        for name in ("index_tip", "middle_tip", "ring_tip"):
            options = {face: face_closest(points[name], cube, face) for face in FACES}
            face, (closest, delta, distance) = min(options.items(), key=lambda item: item[1][2])
            print(f"{name}_closest: face={face} distance={distance:.9f} closest={vector(closest)} delta={vector(delta)}")
        print("physics_steps=0")
        print("lift_test_run=NO")


if __name__ == "__main__":
    main()
