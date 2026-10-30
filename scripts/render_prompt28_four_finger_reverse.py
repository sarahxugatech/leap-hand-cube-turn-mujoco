"""Render the validated four-finger articulated-face reversal."""

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
from validate_four_finger_reverse import REVERSE_DELTAS
from validate_poses import actuator_id, load_pose, object_id, qpos_address, sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_POSE = ROOT / "poses" / "articulated_grasp_R01.json"
RF02_POSE = ROOT / "poses" / "articulated_face_reach_RF02.json"
KF02_POSE = ROOT / "poses" / "articulated_face_followthrough_KF02_R4.json"
KF03_POSE = ROOT / "poses" / "articulated_face_followthrough_KF03.json"
OUTPUT = ROOT / "videos" / "prompt28_four_finger_face_reverse.mp4"
FPS = 30
TITLE = "Prompt 28 - real articulated face / no attachment"


def overlay(
    frame: np.ndarray,
    phase: str,
    angle_deg: float,
    reverse_deg: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=18)
    draw.rounded_rectangle((12, 10, 628, 143), radius=10, fill=(0, 0, 0, 195))
    reversing = phase.startswith("four-finger")
    lines = (
        (TITLE, (255, 255, 255, 255)),
        (f"phase: {phase}", (120, 210, 255, 255)),
        (f"face angle: {angle_deg:+7.3f} deg", (255, 215, 80, 255)),
        (f"reverse progress: {reverse_deg:7.3f} deg", (255, 215, 80, 255)),
        (
            "THUMB + INDEX + MIDDLE + RING MOVING" if reversing else "validated forward setup",
            (110, 235, 140, 255),
        ),
    )
    for index, (line, color) in enumerate(lines):
        draw.text((24, 18 + 24 * index), line, font=font, fill=color)
    return np.asarray(image)


def main() -> None:
    main_before = sha256(MAIN_SCENE)
    scene_before = sha256(SCENE)
    if main_before != EXPECTED_MAIN_SHA256 or scene_before != EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before Prompt 28 render")
    source = json.loads(SOURCE_POSE.read_text(encoding="utf-8"))
    rf02 = json.loads(RF02_POSE.read_text(encoding="utf-8"))
    kf02 = json.loads(KF02_POSE.read_text(encoding="utf-8"))
    kf03 = json.loads(KF03_POSE.read_text(encoding="utf-8"))
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
    model.geom_rgba[core_geom] = (0.10, 0.12, 0.16, 1.0)
    model.geom_rgba[face_geom] = (0.92, 0.20, 0.08, 1.0)
    colors = {
        "if_tip": (1.0, 0.82, 0.05, 1.0),
        "mf_tip": (0.05, 0.85, 1.0, 1.0),
        "rf_tip": (1.0, 0.35, 0.05, 1.0),
        "th_tip": (0.95, 0.15, 0.85, 1.0),
    }
    for name, color in colors.items():
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            model.geom_rgba[geom] = color
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.30
    camera.azimuth = 128
    camera.elevation = -18
    phases = (
        ("stable hold", 500, 30, "hold"),
        ("30 mm lift", 800, 60, "lift"),
        ("forward RF02", 800, 60, "rf02"),
        ("forward KF02", 650, 60, "kf02"),
        ("forward KF03 to ~15.8 deg", 650, 60, "kf03"),
        ("four-finger reverse to zero", 800, 150, "reverse"),
    )
    reverse_start = 0.0
    OUTPUT.parent.mkdir(exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=8) as writer:
            for phase_name, simulation_steps, video_frames, command in phases:
                names = RING_JOINTS if command in {"rf02", "kf02", "kf03"} else tuple(REVERSE_DELTAS)
                starts = {name: float(data.qpos[qpos_address(model, name)]) for name in names}
                if command == "reverse":
                    reverse_start = math.degrees(float(data.qpos[face_qpos]))
                    targets = {
                        name: starts[name] + delta for name, delta in REVERSE_DELTAS.items()
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
                        elif command in {"kf02", "kf03"}:
                            candidate = kf02 if command == "kf02" else kf03
                            steps = int(candidate["search_steps"])
                            if step < steps:
                                blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / steps)
                                for name in RING_JOINTS:
                                    target = float(candidate["target_joints"][name])
                                    data.ctrl[actuator_id(model, name)] = starts[name] + blend * (
                                        target - starts[name]
                                    )
                        elif command == "reverse" and step < 300:
                            blend = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / 300)
                            for name, target in targets.items():
                                data.ctrl[actuator_id(model, name)] = starts[name] + blend * (
                                    target - starts[name]
                                )
                        mujoco.mj_step(model, data)
                    last_step = target_step
                    angle = math.degrees(float(data.qpos[face_qpos]))
                    reverse = max(0.0, reverse_start - angle) if command == "reverse" else 0.0
                    camera.lookat[:] = data.xpos[cube_body]
                    renderer.update_scene(data, camera)
                    writer.append_data(overlay(renderer.render(), phase_name, angle, reverse))
    final_angle = math.degrees(float(data.qpos[face_qpos]))
    reverse_angle = reverse_start - final_angle
    if reverse_angle < 10.0 or abs(final_angle) > 1.0:
        raise RuntimeError(
            f"render did not reproduce validated reversal: final={final_angle}, reverse={reverse_angle}"
        )
    main_after = sha256(MAIN_SCENE)
    scene_after = sha256(SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed while rendering Prompt 28")
    video_hash = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(OUTPUT)
    print(f"duration_s={sum(phase[2] for phase in phases) / FPS:.3f} fps={FPS} resolution=640x480")
    print(f"reverse_start_angle_deg={reverse_start:+.9f}")
    print(f"final_face_angle_deg={final_angle:+.9f}")
    print(f"reverse_angle_deg={reverse_angle:+.9f}")
    print("four_finger_reverse=PASS")
    print(f"video_sha256={video_hash}")
    print(f"main_scene_sha256={main_after}")
    print(f"test_scene_sha256={scene_after}")


if __name__ == "__main__":
    main()
