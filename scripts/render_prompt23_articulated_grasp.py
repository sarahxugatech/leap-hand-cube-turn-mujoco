"""Render the measured articulated-object hold and lift PASS."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from validate_poses import actuator_id, load_pose, object_id, qpos_address, sha256


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCENE = ROOT / "scenes" / "table_cube.xml"
SCENE = ROOT / "scenes" / "table_articulated_face_test.xml"
POSE = ROOT / "poses" / "articulated_grasp_R01.json"
OUTPUT = ROOT / "videos" / "prompt23_articulated_grasp_lift.mp4"
EXPECTED_MAIN_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
EXPECTED_SCENE_SHA256 = "baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb"
EXPECTED_RISE_M = 0.030000466
FPS = 30


def overlay(frame: np.ndarray, phase: str, rise_mm: float, face_deg: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=18)
    draw.rounded_rectangle((14, 12, 626, 143), radius=10, fill=(0, 0, 0, 190))
    lines = (
        ("Prompt 23 - articulated face / free object / no attachment", (255, 255, 255, 255)),
        (f"phase: {phase}", (120, 210, 255, 255)),
        (f"measured core rise: {rise_mm:6.3f} mm", (255, 215, 80, 255)),
        (f"passive face-layer angle: {face_deg:+7.4f} deg", (255, 215, 80, 255)),
        ("HOLD + LIFT: PASS / face turn not attempted", (110, 235, 140, 255)),
    )
    for index, (line, color) in enumerate(lines):
        draw.text((26, 20 + 24 * index), line, font=font, fill=color)
    return np.asarray(image)


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before render")
    payload = json.loads(POSE.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    mount, joints, _, _ = load_pose(POSE, model)
    data = mujoco.MjData(model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_qpos = int(model.jnt_qposadr[face_joint])
    data.qpos[cube_qpos : cube_qpos + 7] = (
        *payload["cube_start_xyz"],
        *payload["cube_start_quat_wxyz"],
    )
    data.qpos[face_qpos] = float(payload["face_turn_rad"])
    for name, value in mount.items() | joints.items():
        data.qpos[qpos_address(model, name)] = value
        data.ctrl[actuator_id(model, name)] = value
    mujoco.mj_forward(model, data)

    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    model.geom_rgba[core_geom] = (0.10, 0.12, 0.16, 1.0)
    model.geom_rgba[face_geom] = (0.92, 0.20, 0.08, 1.0)
    for name in ("if_bs_collision_1", "mf_bs_collision_1", "th_ds_collision_2"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            model.geom_rgba[geom] = (1.0, 0.82, 0.05, 1.0)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[cube_body]
    camera.distance = 0.31
    camera.azimuth = 128
    camera.elevation = -20
    face_start = float(data.qpos[face_qpos])
    lift_start_z: float | None = None
    phases = (
        ("500-step stable hold", 500, 90, "hold"),
        ("30 mm mount lift", 300, 120, "lift"),
        ("500-step lifted hold", 500, 90, "hold"),
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8) as writer:
            for phase_name, simulation_steps, video_frames, command in phases:
                if command == "lift":
                    lift_start_z = float(data.xpos[cube_body, 2])
                last_step = 0
                for frame_index in range(video_frames):
                    target_step = round((frame_index + 1) * simulation_steps / video_frames)
                    for step in range(last_step, target_step):
                        if command == "lift":
                            amount = (step + 1) / simulation_steps
                            blend = 0.5 - 0.5 * math.cos(math.pi * amount)
                            data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
                        mujoco.mj_step(model, data)
                    last_step = target_step
                    rise_mm = 0.0 if lift_start_z is None else (float(data.xpos[cube_body, 2]) - lift_start_z) * 1000.0
                    face_deg = math.degrees(float(data.qpos[face_qpos]) - face_start)
                    camera.lookat[:] = data.xpos[cube_body]
                    renderer.update_scene(data, camera)
                    writer.append_data(overlay(renderer.render(), phase_name, rise_mm, face_deg))

    if lift_start_z is None:
        raise RuntimeError("lift phase did not run")
    final_rise = float(data.xpos[cube_body, 2]) - lift_start_z
    final_face_deg = math.degrees(float(data.qpos[face_qpos]) - face_start)
    if not np.isclose(final_rise, EXPECTED_RISE_M, atol=2e-6):
        raise RuntimeError(f"render replay differs from measured lift: {final_rise}")
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed while rendering Prompt 23")
    video_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(OUTPUT)
    print(f"duration_s={sum(phase[2] for phase in phases) / FPS:.3f} fps={FPS} resolution=640x480")
    print(f"cube_rise_m={final_rise:+.9f}")
    print(f"final_passive_face_angle_deg={final_face_deg:+.9f}")
    print("hold_gate=PASS lift_gate=PASS face_turn_attempted=NO")
    print(f"video_sha256={video_hash}")
    print(f"main_scene_sha256={main_after}")
    print(f"test_scene_sha256={scene_after}")


if __name__ == "__main__":
    main()
