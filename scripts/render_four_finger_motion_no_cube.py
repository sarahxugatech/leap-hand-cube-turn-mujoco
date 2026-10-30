"""Render a cube-free LEAP Hand demo with all four fingers moving."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "menagerie" / "leap_hand" / "scene_right.xml"
OUTPUT_PATH = ROOT / "videos" / "leap_four_finger_motion_no_cube.mp4"
FPS = 30
SIMULATION_STEPS_PER_FRAME = 8

FINGERS = {
    "INDEX": ("if_mcp", "if_rot", "if_pip", "if_dip"),
    "MIDDLE": ("mf_mcp", "mf_rot", "mf_pip", "mf_dip"),
    "RING": ("rf_mcp", "rf_rot", "rf_pip", "rf_dip"),
    "THUMB": ("th_cmc", "th_axl", "th_mcp", "th_ipl"),
}
OPEN = {name: 0.0 for joints in FINGERS.values() for name in joints}
CLOSED = {
    "if_mcp": 1.15,
    "if_rot": 0.15,
    "if_pip": 1.05,
    "if_dip": 0.85,
    "mf_mcp": 1.15,
    "mf_rot": 0.0,
    "mf_pip": 1.05,
    "mf_dip": 0.85,
    "rf_mcp": 1.15,
    "rf_rot": -0.15,
    "rf_pip": 1.05,
    "rf_dip": 0.85,
    "th_cmc": 0.90,
    "th_axl": 0.70,
    "th_mcp": 1.05,
    "th_ipl": 0.75,
}


def blend(start: float, target: float, phase: float) -> float:
    weight = 0.5 - 0.5 * math.cos(math.pi * np.clip(phase, 0.0, 1.0))
    return start + weight * (target - start)


def add_overlay(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=20)
    draw.rounded_rectangle((14, 12, 626, 82), radius=10, fill=(0, 0, 0, 190))
    draw.text((28, 23), "LEAP HAND - NO CUBE", font=font, fill=(255, 255, 255, 255))
    draw.text((28, 51), f"MOVING: {label}", font=font, fill=(110, 225, 255, 255))
    return np.asarray(image)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_qpos = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in OPEN
    }
    actuator = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act")
        for name in OPEN
    }
    if any(value < 0 for value in actuator.values()):
        raise RuntimeError("missing LEAP actuator")

    for name, value in OPEN.items():
        data.ctrl[actuator[name]] = value
    mujoco.mj_forward(model, data)

    tip_colors = {
        "if_tip": (1.0, 0.80, 0.05, 1.0),
        "mf_tip": (0.05, 0.75, 1.0, 1.0),
        "rf_tip": (1.0, 0.30, 0.05, 1.0),
        "th_tip": (0.95, 0.15, 0.85, 1.0),
    }
    for geom_name, color in tip_colors.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            model.geom_rgba[geom_id] = color

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, -0.015, 0.06]
    camera.distance = 0.45
    camera.azimuth = 135
    camera.elevation = -32

    segments = [
        ("OPEN HAND", None, 0.8, False),
        ("INDEX", "INDEX", 1.0, True),
        ("INDEX", "INDEX", 0.8, False),
        ("MIDDLE", "MIDDLE", 1.0, True),
        ("MIDDLE", "MIDDLE", 0.8, False),
        ("RING", "RING", 1.0, True),
        ("RING", "RING", 0.8, False),
        ("THUMB", "THUMB", 1.0, True),
        ("THUMB", "THUMB", 0.8, False),
        ("ALL FOUR FINGERS", "ALL", 1.3, True),
        ("ALL FOUR FINGERS", "ALL", 1.0, False),
        ("OPEN HAND", None, 0.7, False),
    ]
    observed = {finger: [] for finger in FINGERS}
    current_targets = dict(OPEN)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(OUTPUT_PATH, fps=FPS, codec="libx264", quality=8) as writer:
            for label, selected, duration, closing in segments:
                start_targets = dict(current_targets)
                end_targets = dict(OPEN)
                active = tuple(OPEN) if selected == "ALL" else FINGERS.get(selected, ())
                if closing:
                    for name in active:
                        end_targets[name] = CLOSED[name]
                frame_count = round(duration * FPS)
                for frame_index in range(frame_count):
                    phase = (frame_index + 1) / frame_count
                    for name in OPEN:
                        target = blend(start_targets[name], end_targets[name], phase)
                        data.ctrl[actuator[name]] = target
                    for _ in range(SIMULATION_STEPS_PER_FRAME):
                        mujoco.mj_step(model, data)
                    for finger, joints in FINGERS.items():
                        observed[finger].append([float(data.qpos[joint_qpos[name]]) for name in joints])
                    renderer.update_scene(data, camera)
                    writer.append_data(add_overlay(renderer.render(), label))
                current_targets = end_targets

    ranges = {}
    for finger, values in observed.items():
        values_array = np.asarray(values)
        ranges[finger] = float(np.max(np.ptp(values_array, axis=0)))
        if ranges[finger] < 0.5:
            raise RuntimeError(f"{finger} did not visibly move: range={ranges[finger]:.3f} rad")

    video_hash = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    duration = sum(segment[2] for segment in segments)
    print(OUTPUT_PATH)
    print(f"duration_s={duration:.1f} fps={FPS} resolution=640x480")
    print("cube_present=NO")
    print("four_fingers_moved=PASS")
    for finger, movement_range in ranges.items():
        print(f"{finger.lower()}_max_joint_range_rad={movement_range:.6f}")
    print(f"video_sha256={video_hash}")


if __name__ == "__main__":
    main()
