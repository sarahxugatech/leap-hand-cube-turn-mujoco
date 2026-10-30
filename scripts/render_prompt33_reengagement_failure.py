"""Render the best Prompt 33 re-engagement failure without hiding the loss."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from search_articulated_face_followthrough import (
    EXPECTED_MAIN_SHA256,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    RING_JOINTS,
    SCENE,
)
from validate_poses import actuator_id, load_pose, object_id, qpos_address, sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_POSE = ROOT / "poses" / "articulated_grasp_R01.json"
RF02_POSE = ROOT / "poses" / "articulated_face_reach_RF02.json"
KF02_POSE = ROOT / "poses" / "articulated_face_followthrough_KF02_R4.json"
OUTPUT = ROOT / "videos" / "prompt33_reengagement_geometry_failure.mp4"
FPS = 30


def overlay(
    frame: np.ndarray,
    phase: str,
    angle_deg: float,
    force_n: float,
    contact_rate: float,
    drift_mm: np.ndarray,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=17)
    draw.rounded_rectangle((12, 10, 628, 166), radius=10, fill=(0, 0, 0, 195))
    failure = phase.startswith("re-engagement")
    lines = (
        ("Prompt 33 - real MuJoCo replay / no attachment", (255, 255, 255, 255)),
        (f"phase: {phase}", (120, 210, 255, 255)),
        (f"face angle: {angle_deg:+7.3f} deg", (255, 215, 80, 255)),
        (f"ring/face force: {force_n:6.3f} N", (255, 215, 80, 255)),
        (f"transition contact rate: {100 * contact_rate:5.1f}%", (255, 215, 80, 255)),
        (
            f"max drift XYZ: {drift_mm[0]:.3f}, {drift_mm[1]:.3f}, {drift_mm[2]:.3f} mm",
            (255, 215, 80, 255),
        ),
        ("RE-ENGAGEMENT: FAIL (finger remains detached)" if failure else "validated setup", (255, 100, 90, 255) if failure else (110, 235, 140, 255)),
    )
    for index, (line, color) in enumerate(lines):
        draw.text((24, 18 + 21 * index), line, font=font, fill=color)
    return np.asarray(image)


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before Prompt 33 render")
    source = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    rf02 = json.loads(RF02_POSE.read_text(encoding="utf-8"))
    kf02 = json.loads(KF02_POSE.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    mount, joints, _, _ = load_pose(SOURCE_POSE, model)
    data = mujoco.MjData(model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube_qpos = int(model.jnt_qposadr[cube_joint])
    face_qpos = int(model.jnt_qposadr[face_joint])
    data.qpos[cube_qpos : cube_qpos + 7] = (
        *source["cube_start_xyz"],
        *source["cube_start_quat_wxyz"],
    )
    data.qpos[face_qpos] = float(source["face_turn_rad"])
    for name, value in mount.items() | joints.items():
        data.qpos[qpos_address(model, name)] = value
        data.ctrl[actuator_id(model, name)] = value
    mujoco.mj_forward(model, data)

    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    rf_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_ds_collision_1")
    model.geom_rgba[core_geom] = (0.10, 0.12, 0.16, 1.0)
    model.geom_rgba[face_geom] = (0.92, 0.20, 0.08, 1.0)
    model.geom_rgba[rf_geom] = (1.0, 0.82, 0.05, 1.0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.29
    camera.azimuth = 128
    camera.elevation = -18

    current_force = 0.0
    transition_contacts = 0
    transition_samples = 0
    maximum_drift = np.zeros(3)
    candidate_cube_start: np.ndarray | None = None

    def measure_candidate(transition: bool) -> None:
        nonlocal current_force, transition_contacts, transition_samples, maximum_drift
        current_force = 0.0
        for contact_index, contact in enumerate(data.contact[: data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            if face_geom not in pair or rf_geom not in pair:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            current_force += max(0.0, float(force[0]))
        if transition:
            transition_samples += 1
            transition_contacts += current_force > 0.001
        if candidate_cube_start is not None:
            maximum_drift = np.maximum(
                maximum_drift,
                1000.0 * np.abs(data.xpos[cube_body] - candidate_cube_start),
            )

    phases = (
        ("stable hold", 500, 30, "hold"),
        ("30 mm lift", 800, 45, "lift"),
        ("RF02 reach", 800, 45, "rf02"),
        ("KF02 to 9.5 deg", 650, 45, "kf02"),
        ("re-engagement attempt - measured FAIL", 650, 120, "candidate"),
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8) as writer:
            for phase_name, simulation_steps, video_frames, command in phases:
                starts = {
                    name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS
                }
                if command == "candidate":
                    candidate_cube_start = data.xpos[cube_body].copy()
                last_step = 0
                for frame_index in range(video_frames):
                    target_step = round((frame_index + 1) * simulation_steps / video_frames)
                    for step in range(last_step, target_step):
                        if command == "lift":
                            phase = min((step + 1) / 300, 1.0)
                            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
                            data.ctrl[actuator_id(model, "mount_z")] = mount["mount_z"] + 0.030 * blend
                        elif command == "rf02" and step < int(rf02["transition_steps"]):
                            blend = 0.5 - 0.5 * math.cos(
                                math.pi * (step + 1) / int(rf02["transition_steps"])
                            )
                            for name in RING_JOINTS:
                                target = float(rf02["target_joints"][name])
                                data.ctrl[actuator_id(model, name)] = joints[name] + blend * (
                                    target - joints[name]
                                )
                        elif command == "kf02" and step < int(kf02["search_steps"]):
                            blend = 0.5 - 0.5 * math.cos(
                                math.pi * (step + 1) / int(kf02["search_steps"])
                            )
                            for name in RING_JOINTS:
                                target = float(kf02["target_joints"][name])
                                data.ctrl[actuator_id(model, name)] = starts[name] + blend * (
                                    target - starts[name]
                                )
                        elif command == "candidate" and step < 150:
                            dip_blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / 150)
                            push_blend = 0.0
                            if step >= 30:
                                push_blend = 0.5 - 0.5 * math.cos(
                                    math.pi * (step - 30 + 1) / 120
                                )
                            data.ctrl[actuator_id(model, "rf_dip")] = starts["rf_dip"] + 0.04 * dip_blend
                            data.ctrl[actuator_id(model, "rf_mcp")] = starts["rf_mcp"] + 0.10 * push_blend
                            data.ctrl[actuator_id(model, "rf_pip")] = starts["rf_pip"] - 0.06 * push_blend
                            data.ctrl[actuator_id(model, "rf_rot")] = starts["rf_rot"]
                        mujoco.mj_step(model, data)
                        if command == "candidate":
                            measure_candidate(step < 150)
                    last_step = target_step
                    camera.lookat[:] = data.xpos[cube_body]
                    renderer.update_scene(data, camera)
                    angle = math.degrees(float(data.qpos[face_qpos]))
                    rate = transition_contacts / transition_samples if transition_samples else 0.0
                    writer.append_data(
                        overlay(
                            renderer.render(),
                            phase_name,
                            angle,
                            current_force if command == "candidate" else 0.0,
                            rate,
                            maximum_drift,
                        )
                    )

    final_angle = math.degrees(float(data.qpos[face_qpos]))
    contact_rate = transition_contacts / transition_samples
    if contact_rate >= 0.85 or final_angle > 13.5:
        raise RuntimeError("render unexpectedly escaped the measured failure category")
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed while rendering Prompt 33")
    video_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(OUTPUT)
    print(f"duration_s={sum(phase[2] for phase in phases) / FPS:.3f} fps={FPS} resolution=640x480")
    print(f"final_face_angle_deg={final_angle:+.9f}")
    print(f"transition_contact_rate={contact_rate:.9f}")
    print(f"maximum_axis_drift_mm={maximum_drift.tolist()}")
    print("reengagement_gate=FAIL")
    print(f"video_sha256={video_hash}")
    print(f"main_scene_sha256={main_after}")
    print(f"test_scene_sha256={scene_after}")


if __name__ == "__main__":
    main()
