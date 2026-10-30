"""Render the four mount-range-compatible Prompt 07 poses without simulation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from import_leap_cache_pose import (
    CUBE_CENTRE,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    MENAGERIE_ORDER,
    ROOT,
    cube_symmetries,
    free_mount_scene,
    quaternion_wxyz,
    scan_orientation,
    sha256,
)
from import_mjlab_left5_pose import (
    LEFT_FROM_RIGHT,
    LEFT_FROM_RIGHT_TRANSLATION,
    MIRRORED_JOINTS,
    SOURCE_ENTITY_POSITION,
)


CACHE = Path(
    "/private/tmp/in_hand_rotation_mjlab_prompt06/src/in_hand_rotation_mjlab/"
    "tasks/hand_cube/cache/leap_left_custom_grasp_cache_cube5cm.npz"
)
OUTPUT = ROOT / "videos" / "prompt07_static_rejection.mp4"
ORIENTATIONS = (1, 2, 8, 11)
FPS = 30
SECONDS_PER_POSE = 2


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints: dict[str, float],
    position: np.ndarray,
    rotation: np.ndarray,
) -> None:
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    data.qpos[cube_qpos : cube_qpos + 7] = (*CUBE_CENTRE, 1, 0, 0, 0)
    mount_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "prompt06_mount_free")
    mount_qpos = int(model.jnt_qposadr[mount_joint])
    data.qpos[mount_qpos : mount_qpos + 7] = (
        *position,
        *quaternion_wxyz(rotation),
    )
    for name, value in joints.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint])] = value
    mujoco.mj_forward(model, data)


def add_overlay(frame: np.ndarray, orientation: int, distances: dict[str, float]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=18)
    draw.rounded_rectangle((16, 14, 624, 132), radius=10, fill=(0, 0, 0, 185))
    lines = (
        "Prompt 07 - static safety rejection",
        f"orientation {orientation} | table penetration: {-1000 * distances['hand_table_m']:.2f} mm",
        f"cube penetration: {-1000 * distances['hand_cube_m']:.2f} mm",
        "REJECTED - no dynamics / grasp / lift / rotation",
    )
    colors = ((255, 255, 255, 255), (255, 210, 80, 255), (255, 210, 80, 255), (255, 100, 100, 255))
    for line_index, (line, color) in enumerate(zip(lines, colors, strict=True)):
        draw.text((28, 24 + 26 * line_index), line, fill=color, font=font)
    return np.asarray(image)


def main() -> None:
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    if not CACHE.exists():
        raise FileNotFoundError(CACHE)

    with np.load(CACHE, allow_pickle=False) as cache:
        names = [str(name) for name in cache["joint_names"]]
        if names != list(MENAGERIE_ORDER):
            raise ValueError(f"unexpected joint order: {names}")
        row_index = int(np.argmin(np.abs(cache["cube_size"] - 0.025)))
        source_joints = cache["joint_pos"][row_index].astype(float)
        source_cube = cache["cube_pose_rel"][row_index].astype(float)

    joints = {
        name: float(-value if name in MIRRORED_JOINTS else value)
        for name, value in zip(names, source_joints, strict=True)
    }
    right_cube_position = LEFT_FROM_RIGHT @ (
        source_cube[:3] - SOURCE_ENTITY_POSITION - LEFT_FROM_RIGHT_TRANSLATION
    )
    symmetries = cube_symmetries()

    OUTPUT.parent.mkdir(exist_ok=True)
    source = MAIN_SCENE.read_text(encoding="utf-8")
    with TemporaryDirectory(prefix="prompt07_video_", dir=ROOT / "logs") as temporary:
        scene = free_mount_scene(source, Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [-0.01, 0.0, 0.085]
        camera.distance = 0.42
        camera.elevation = -22

        with mujoco.Renderer(model, height=480, width=640) as renderer:
            with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8) as writer:
                for orientation in ORIENTATIONS:
                    rotation = symmetries[orientation]
                    position = CUBE_CENTRE - rotation @ right_cube_position
                    distances = scan_orientation(model, joints, position, rotation)
                    set_pose(model, data, joints, position, rotation)
                    for frame_index in range(FPS * SECONDS_PER_POSE):
                        camera.azimuth = 118 + 18 * frame_index / (FPS * SECONDS_PER_POSE - 1)
                        renderer.update_scene(data, camera)
                        writer.append_data(add_overlay(renderer.render(), orientation, distances))

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during video rendering")
    video_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(OUTPUT)
    print(f"source_row={row_index}")
    print(f"duration_s={len(ORIENTATIONS) * SECONDS_PER_POSE} fps={FPS} resolution=640x480")
    print(f"video_sha256={video_hash}")
    print(f"main_scene_sha256={after_hash}")


if __name__ == "__main__":
    main()
