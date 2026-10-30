"""Render the real Prompt 27 closest KF03_R1 result and its strict failure."""

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
OUTPUT = ROOT / "videos" / "prompt27_KF03_R1_strict_failure.mp4"
FPS = 30


def overlay(
    frame: np.ndarray,
    phase: str,
    face_deg: float,
    force_n: float,
    max_force_n: float,
    drift_mm: np.ndarray,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=17)
    draw.rounded_rectangle((12, 10, 628, 166), radius=10, fill=(0, 0, 0, 195))
    final_phase = phase.startswith("KF03_R1")
    verdict = "STRICT GATE: FAIL" if final_phase else "measured scripted trajectory"
    color = (255, 100, 90, 255) if final_phase else (110, 235, 140, 255)
    lines = (
        ("Prompt 27 - real articulated face / no attachment", (255, 255, 255, 255)),
        (f"phase: {phase}", (120, 210, 255, 255)),
        (f"face angle: {face_deg:+7.3f} deg", (255, 215, 80, 255)),
        (f"ring normal force: {force_n:6.3f} N   peak: {max_force_n:6.3f} N", (255, 215, 80, 255)),
        (f"cube axis drift XYZ: {drift_mm[0]:.3f}, {drift_mm[1]:.3f}, {drift_mm[2]:.3f} mm", (255, 215, 80, 255)),
        (verdict, color),
    )
    for index, (line, line_color) in enumerate(lines):
        draw.text((24, 18 + 24 * index), line, font=font, fill=line_color)
    return np.asarray(image)


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before Prompt 27 render")
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
    face_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "face_layer_geom")
    core_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_core_geom")
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
    maximum_force = 0.0
    maximum_axis_drift = np.zeros(3)
    r1_cube_start: np.ndarray | None = None

    def measure_r1() -> None:
        nonlocal current_force, maximum_force, maximum_axis_drift
        current_force = 0.0
        for contact_index, contact in enumerate(data.contact[: data.ncon]):
            first, second = int(contact.geom1), int(contact.geom2)
            if face_geom not in {first, second} or rf_geom not in {first, second}:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            current_force += max(0.0, float(force[0]))
        maximum_force = max(maximum_force, current_force)
        if r1_cube_start is not None:
            maximum_axis_drift = np.maximum(
                maximum_axis_drift,
                np.abs(data.xpos[cube_body] - r1_cube_start) * 1000.0,
            )

    phases: list[tuple[str, int, int, object]] = []
    phases.append(("stable hold", 500, 45, None))
    phases.append(("30 mm lift + lifted hold", 800, 75, "lift"))
    phases.append(("RF02 reach to ~3.4 deg", 800, 75, "rf02"))
    phases.append(("KF02 follow-through to ~9.5 deg", 650, 75, "kf02"))
    phases.append(("KF03_R1 closest result - strict fail", 650, 90, "r1"))
    OUTPUT.parent.mkdir(exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8) as writer:
            for phase_name, simulation_steps, video_frames, command in phases:
                phase_start = {
                    name: float(data.qpos[qpos_address(model, name)]) for name in RING_JOINTS
                }
                if command == "r1":
                    # Match the search branch restore path, which calls mj_forward before evaluation.
                    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
                    state = np.empty(mujoco.mj_stateSize(model, state_spec))
                    mujoco.mj_getState(model, data, state, state_spec)
                    mujoco.mj_setState(model, data, state, state_spec)
                    mujoco.mj_forward(model, data)
                    r1_cube_start = data.xpos[cube_body].copy()
                    r1_targets = {
                        "rf_mcp": phase_start["rf_mcp"] + 0.06,
                        "rf_rot": float(kf02["target_joints"]["rf_rot"]),
                        "rf_pip": phase_start["rf_pip"] - 0.06,
                        "rf_dip": phase_start["rf_dip"] - 0.06,
                    }
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
                                data.ctrl[actuator_id(model, name)] = phase_start[name] + blend * (
                                    target - phase_start[name]
                                )
                        elif command == "r1" and step < 150:
                            blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / 150)
                            for name in RING_JOINTS:
                                data.ctrl[actuator_id(model, name)] = phase_start[name] + blend * (
                                    r1_targets[name] - phase_start[name]
                                )
                        mujoco.mj_step(model, data)
                        if command == "r1":
                            measure_r1()
                    last_step = target_step
                    camera.lookat[:] = data.xpos[cube_body]
                    renderer.update_scene(data, camera)
                    face_deg = math.degrees(float(data.qpos[face_qpos]))
                    writer.append_data(
                        overlay(
                            renderer.render(),
                            phase_name,
                            face_deg,
                            current_force if command == "r1" else 0.0,
                            maximum_force,
                            maximum_axis_drift,
                        )
                    )

    final_angle = math.degrees(float(data.qpos[face_qpos]))
    if not 15.0 <= final_angle <= 20.0:
        raise RuntimeError(f"render replay left the measured angle band: {final_angle}")
    if maximum_force <= 0.35:
        raise RuntimeError(f"render replay did not reproduce the force failure: {maximum_force}")
    if float(np.max(maximum_axis_drift)) <= 0.3:
        raise RuntimeError(
            f"render replay did not reproduce the drift failure: {maximum_axis_drift}"
        )
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed while rendering Prompt 27")
    video_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(OUTPUT)
    print(f"duration_s={sum(phase[2] for phase in phases) / FPS:.3f} fps={FPS} resolution=640x480")
    print(f"final_face_angle_deg={final_angle:+.9f}")
    print(f"maximum_normal_force_n={maximum_force:.9f}")
    print(f"maximum_axis_drift_mm={maximum_axis_drift.tolist()}")
    print("strict_gate=FAIL")
    print(f"video_sha256={video_hash}")
    print(f"main_scene_sha256={main_after}")
    print(f"test_scene_sha256={scene_after}")


if __name__ == "__main__":
    main()
