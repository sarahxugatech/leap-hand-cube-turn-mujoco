"""Render the measured Prompt 16 oriented grasp, lift, and small torque probe."""

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


POSE = ROOT / "poses" / "oriented_grasp_O01.json"
OUTPUT = ROOT / "videos" / "prompt16_oriented_grasp_torque_probe.mp4"
FPS = 30


def quaternion_angle_degrees(start: np.ndarray, end: np.ndarray) -> float:
    delta = np.empty(3)
    mujoco.mju_subQuat(delta, end, start)
    return math.degrees(float(np.linalg.norm(delta)))


def overlay(frame: np.ndarray, phase: str, angle: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=18)
    draw.rounded_rectangle((14, 12, 626, 119), radius=10, fill=(0, 0, 0, 185))
    lines = (
        ("Prompt 16 - TEST scene / free cube / no attachment", (255, 255, 255, 255)),
        (f"phase: {phase}", (120, 210, 255, 255)),
        (f"measured probe rotation: {angle:6.3f} deg", (255, 215, 80, 255)),
        ("small torque probe only - NOT a 90 deg success", (255, 130, 110, 255)),
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

    with TemporaryDirectory(prefix="prompt16_video_", dir=ROOT / "logs") as temporary:
        scene = test_scene_copy(MAIN_SCENE.read_text(), Path(temporary), enable_yaw=True)
        model = mujoco.MjModel.from_xml_path(str(scene))
        mount, joints, yaw_degrees, _ = load_pose(POSE, model)
        data = mujoco.MjData(model)
        cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        data.qpos[cube_qpos : cube_qpos + 7] = (*payload["cube_start_xyz"], *payload["cube_start_quat_wxyz"])
        for name, value in mount.items() | joints.items():
            data.qpos[qpos_address(model, name)] = value
            data.ctrl[actuator_id(model, name)] = value
        yaw = math.radians(yaw_degrees)
        data.qpos[qpos_address(model, "mount_yaw")] = yaw
        data.ctrl[actuator_id(model, "mount_yaw")] = yaw
        mujoco.mj_forward(model, data)

        cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        model.geom_rgba[cube_geom] = (0.90, 0.22, 0.07, 1.0)
        for name in ("rf_md_collision_5", "mf_px_collision", "th_ds_collision_1", "th_ds_collision_2"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom >= 0:
                model.geom_rgba[geom] = (1.0, 0.82, 0.05, 1.0)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = payload["cube_start_xyz"]
        camera.distance = 0.30
        camera.azimuth = 132
        camera.elevation = -18
        probe_start_quaternion: np.ndarray | None = None
        probe_start_position: np.ndarray | None = None

        phases = (
            ("stable oriented grasp", 200, 60, None),
            ("30 mm mount lift", 300, 90, "lift"),
            ("lifted hold", 500, 45, None),
            ("rf_rot -0.10 rad torque probe", 150, 120, "probe"),
            ("post-probe hold", 100, 75, None),
        )
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8) as writer:
                for phase_name, simulation_steps, video_frames, command in phases:
                    if command == "probe":
                        probe_start_quaternion = data.xquat[cube_body].copy()
                        probe_start_position = data.xpos[cube_body].copy()
                    last_step = 0
                    for frame_index in range(video_frames):
                        target_step = round((frame_index + 1) * simulation_steps / video_frames)
                        for step in range(last_step, target_step):
                            phase = (step + 1) / simulation_steps
                            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                            if command == "lift":
                                data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
                            elif command == "probe":
                                data.ctrl[actuator_id(model, "rf_rot")] = joints["rf_rot"] - 0.10 * blend
                            mujoco.mj_step(model, data)
                        last_step = target_step
                        angle = (
                            quaternion_angle_degrees(probe_start_quaternion, data.xquat[cube_body])
                            if probe_start_quaternion is not None else 0.0
                        )
                        camera.lookat[:] = data.xpos[cube_body]
                        renderer.update_scene(data, camera)
                        writer.append_data(overlay(renderer.render(), phase_name, angle))

        if probe_start_quaternion is None or probe_start_position is None:
            raise RuntimeError("probe phase did not run")
        final_angle = quaternion_angle_degrees(probe_start_quaternion, data.xquat[cube_body])
        maximum_visible_displacement = float(np.linalg.norm(data.xpos[cube_body] - probe_start_position))

    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed while rendering Prompt 16")
    video_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(OUTPUT)
    print(f"duration_s={sum(phase[2] for phase in phases) / FPS:.3f} fps={FPS} resolution=640x480")
    print(f"final_probe_rotation_deg={final_angle:.9f}")
    print(f"final_probe_displacement_m={maximum_visible_displacement:.9f}")
    print(f"video_sha256={video_hash}")
    print(f"main_scene_sha256={after_hash}")


if __name__ == "__main__":
    main()
