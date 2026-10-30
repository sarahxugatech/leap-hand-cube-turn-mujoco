"""Render the independently measured Prompt 19 Q01 lift-gate failure."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from validate_poses import (
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    ROOT,
    actuator_id,
    load_pose,
    object_id,
    qpos_address,
    sha256,
    test_scene_copy,
)


POSE = ROOT / "poses" / "fixed_base_axis_Q01.json"
OUTPUT = ROOT / "videos" / "prompt19_Q01_lift_gate_failure.mp4"
TITLE = "Prompt 19 Q01"
EXPECTED_RISE_M = 0.016788440
FPS = 30
LIFT_M = 0.030


def overlay(frame: np.ndarray, phase: str, cube_rise_mm: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=18)
    draw.rounded_rectangle((14, 12, 626, 143), radius=10, fill=(0, 0, 0, 190))
    lines = (
        (f"{TITLE} - TEST scene / free cube / no attachment", (255, 255, 255, 255)),
        (f"phase: {phase}", (120, 210, 255, 255)),
        ("commanded hand lift: 30.000 mm", (255, 215, 80, 255)),
        (f"measured cube rise: {cube_rise_mm:6.3f} mm", (255, 215, 80, 255)),
        ("LIFT GATE: FAIL (requires >= 20 mm) / no rotation", (255, 115, 100, 255)),
    )
    for index, (line, color) in enumerate(lines):
        draw.text((26, 20 + 24 * index), line, font=font, fill=color)
    return np.asarray(image)


def main() -> None:
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    payload = json.loads(POSE.read_text())
    OUTPUT.parent.mkdir(exist_ok=True)

    with TemporaryDirectory(prefix="prompt19_video_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(
            MAIN_SCENE.read_text(),
            Path(temporary),
            enable_yaw=True,
            mount_quat=payload["mount_quat_wxyz"],
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, joints, yaw_degrees, _ = load_pose(POSE, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        data.qpos[cube_qpos : cube_qpos + 7] = (
            *payload["cube_start_xyz"], *payload["cube_start_quat_wxyz"],
        )
        for name, value in mount.items() | joints.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        yaw = math.radians(yaw_degrees)
        data.qpos[qpos_address(model, "mount_yaw")] = yaw
        data.ctrl[actuator_id(model, "mount_yaw")] = yaw
        mujoco.mj_forward(model, data)

        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        cube_start = data.xpos[cube_body].copy()
        model.geom_rgba[cube_geom] = (0.90, 0.22, 0.07, 1.0)
        for name in ("rf_md_collision_5", "mf_px_collision", "th_tip"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom >= 0:
                model.geom_rgba[geom] = (1.0, 0.82, 0.05, 1.0)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = cube_start + np.array([0.0, 0.0, 0.005])
        camera.distance = 0.31
        camera.azimuth = 122
        camera.elevation = -17
        phases = (
            ("30 mm mount lift", 300, 120, True),
            ("500-step post-lift hold", 500, 150, False),
        )
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8) as writer:
                for phase_name, simulation_steps, video_frames, lifting in phases:
                    last_step = 0
                    for frame_index in range(video_frames):
                        target_step = round((frame_index + 1) * simulation_steps / video_frames)
                        for step in range(last_step, target_step):
                            if lifting:
                                phase = (step + 1) / simulation_steps
                                blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                                data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + LIFT_M * blend
                            mujoco.mj_step(model, data)
                        last_step = target_step
                        cube_rise_mm = float(data.xpos[cube_body, 2] - cube_start[2]) * 1000.0
                        renderer.update_scene(data, camera)
                        writer.append_data(overlay(renderer.render(), phase_name, cube_rise_mm))

        final_rise = float(data.xpos[cube_body, 2] - cube_start[2])
        if not np.isclose(final_rise, EXPECTED_RISE_M, atol=1e-6):
            raise RuntimeError(f"render replay differs from measured lift: {final_rise}")

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed while rendering Prompt 19")
    video_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(OUTPUT)
    print(f"duration_s={sum(phase[2] for phase in phases) / FPS:.3f} fps={FPS} resolution=640x480")
    print(f"cube_rise_m={final_rise:+.9f}")
    print("lift_gate=FAIL")
    print(f"video_sha256={video_hash}")
    print(f"main_scene_sha256={after_hash}")


if __name__ == "__main__":
    main()
