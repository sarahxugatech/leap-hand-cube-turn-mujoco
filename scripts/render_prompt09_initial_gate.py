"""Render the rejected Prompt 09 initial-contact state without stepping physics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from validate_mjlab_left_source_hold import (
    CACHE,
    EXPECTED_MAIN_SHA,
    JOINT_NAMES,
    MAIN_SCENE,
    ROOT,
    ROW,
    sha256,
    test_scene,
)


OUTPUT = ROOT / "videos" / "prompt09_initial_gate_failure.mp4"
FPS = 30
DURATION_SECONDS = 6


def overlay(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=18)
    draw.rounded_rectangle((16, 14, 624, 142), radius=10, fill=(0, 0, 0, 190))
    lines = (
        ("Prompt 09 - original modified-left geometry", (255, 255, 255, 255)),
        ("initial contacts: palm + middle-finger base", (255, 210, 80, 255)),
        ("thumb: NO | non-thumb fingers: 1", (255, 100, 100, 255)),
        ("INITIAL GATE: FAIL", (255, 100, 100, 255)),
        ("STATIC ONLY - 0 dynamics steps / no lift / no rotation", (255, 255, 255, 255)),
    )
    for index, (line, color) in enumerate(lines):
        draw.text((28, 23 + 24 * index), line, font=font, fill=color)
    return np.asarray(image)


def main() -> None:
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_MAIN_SHA:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    with np.load(CACHE, allow_pickle=False) as cache:
        cube_size = float(cache["cube_size"][ROW])
        joint_values = cache["joint_pos"][ROW].astype(float)
        cube_pose = cache["cube_pose_rel"][ROW].astype(float)

    OUTPUT.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="prompt09_video_", dir=ROOT / "logs") as temporary:
        scene = Path(temporary) / "source_left_hold.xml"
        test_scene(scene, cube_size, cube_pose)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        for joint_name, value in zip(JOINT_NAMES, joint_values, strict=True):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_act")
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.ctrl[actuator] = value
        model.geom_friction[:, 0] = 1.0
        mujoco.mj_forward(model, data)

        cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        model.geom_rgba[cube_geom] = (0.9, 0.18, 0.08, 1.0)
        for geom_name in ("palm_collision_6", "mf_bs_collision_1"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            model.geom_rgba[geom] = (1.0, 0.85, 0.05, 1.0)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = cube_pose[:3]
        camera.distance = 0.36
        camera.elevation = -18
        frames = FPS * DURATION_SECONDS
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8) as writer:
                for frame in range(frames):
                    camera.azimuth = 90 + 100 * frame / (frames - 1)
                    renderer.update_scene(data, camera)
                    writer.append_data(overlay(renderer.render()))

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed while rendering Prompt 09")
    video_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(OUTPUT)
    print(f"duration_s={DURATION_SECONDS} fps={FPS} resolution=640x480")
    print(f"video_sha256={video_hash}")
    print(f"main_scene_sha256={after_hash}")


if __name__ == "__main__":
    main()
